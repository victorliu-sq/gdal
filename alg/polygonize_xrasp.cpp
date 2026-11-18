/******************************************************************************
 * Project:  GDAL
 * Purpose:  GDALPolygonize() with X-Rasp, a parallel polygonizer
 *
 ******************************************************************************
 * SPDX-License-Identifier: MIT
 ****************************************************************************/

// X-Rasp finds the polygons of a raster with all the threads of the machine:
// every vertex of every outline is found on its own, finds its next vertex with
// a search, and the outlines are ordered by list ranking.  The features are
// those of the default polygonizer, in the same order: the same polygons, their
// holes in the same order, every ring from the same vertex.  The algorithm is
// under alg/xrasp, a copy of the CPU version of the X-Rasp library (README.md
// there).  This file is what connects it to GDAL: the band as a whole in
// memory (where it is, for a MEM band; read otherwise), polygonized, and the
// polygons written to the layer as the features OGRPolygonWriter writes.  The
// feature objects are made by the calling thread while the other threads
// still find the order of the polygons; their points are filled in by all.

#include "polygonize_xrasp.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include "cpl_conv.h"
#include "cpl_multiproc.h"
#include "cpl_string.h"
#include "cpl_vsi.h"
#include "gdal_alg_priv.h"
#include "gdal_priv.h"
#include "cpl_worker_thread_pool.h"
#include "gdal_thread_pool.h"
#include "memdataset.h"
#include "ogr_feature.h"
#include "ogr_geometry.h"
#include "ogrsf_frmts.h"

#include "xrasp/core/polygonize_whole.h"
#include "xrasp/util/cpu_threads.h"

namespace
{

/** The threads of X-Rasp where GDAL does not use OpenMP: GDAL's worker threads
 * (xrasp/util/cpu_threads.h).  pfnPart(pJob, i) for i = 0 .. nParts-1, the
 * first on the calling thread, the others on the pool; returns when all are
 * done.  A part the pool does not take runs on the calling thread. */
void RunOnGDALPool(int nParts, xrasp::CpuPartFn pfnPart, void *pJob)
{
    CPLWorkerThreadPool *poPool = GDALGetGlobalThreadPool(nParts - 1);
    CPLJobQueuePtr poQueue = poPool ? poPool->CreateJobQueue() : nullptr;
    for (int iPart = 1; iPart < nParts; ++iPart)
    {
        bool bSubmitted = false;
        if (poQueue)
        {
            try
            {
                bSubmitted = poQueue->SubmitJob([pfnPart, pJob, iPart]()
                                                { pfnPart(pJob, iPart); });
            }
            catch (const std::exception &)
            {
            }
        }
        if (!bSubmitted)
            pfnPart(pJob, iPart);
    }
    pfnPart(pJob, 0);
    if (poQueue)
        poQueue->WaitCompletion();
}

/** Number of physical cores: X-Rasp is bound by memory, a second thread on a
 * core does not help it. */
int PhysicalCores()
{
    const int nLogical = CPLGetNumCPUs();
#ifdef __linux__
    std::set<std::pair<int, int>> oCores;
    for (int iCPU = 0; iCPU < nLogical; ++iCPU)
    {
        const std::string osDir = "/sys/devices/system/cpu/cpu" +
                                  std::to_string(iCPU) + "/topology/";
        std::ifstream oPackage(osDir + "physical_package_id");
        std::ifstream oCore(osDir + "core_id");
        int nPackage = -1;
        int nCore = -1;
        if (oPackage >> nPackage && oCore >> nCore)
            oCores.insert({nPackage, nCore});
    }
    if (!oCores.empty() && static_cast<int>(oCores.size()) <= nLogical)
        return static_cast<int>(oCores.size());
#endif
    return nLogical > 0 ? nLogical : 1;
}

/** GDAL_NUM_THREADS when it is set, the physical cores otherwise. */
int ThreadCount()
{
    const char *pszThreads = CPLGetConfigOption("GDAL_NUM_THREADS", nullptr);
    if (pszThreads)
    {
        if (EQUAL(pszThreads, "ALL_CPUS"))
            return CPLGetNumCPUs();
        const int nThreads = atoi(pszThreads);
        if (nThreads >= 1)
            return nThreads;
    }
    return PhysicalCores();
}

double Seconds(std::chrono::steady_clock::time_point oSince)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         oSince)
        .count();
}

/** One polygon as a feature of the layer, as OGRPolygonWriter makes it, its
 * rings of the right length with nothing in them yet. */
std::unique_ptr<OGRFeature> EmptyFeatureOf(const xrasp::Polygons &oPolygons,
                                           int64_t iPoly,
                                           OGRFeatureDefn *poDefn,
                                           int iPixValField, double dfValue)
{
    // In the CPU build "device" memory is host memory.
    const int64_t *panPolyOffset = oPolygons.poly_offsets.data();
    const int64_t *panRingOffset = oPolygons.ring_offsets.data();
    auto poPolygon = std::make_unique<OGRPolygon>();
    // the outer ring first
    for (int64_t iRing = panPolyOffset[iPoly]; iRing < panPolyOffset[iPoly + 1];
         ++iRing)
    {
        // a ring is closed with its first point
        const int nPoints =
            static_cast<int>(panRingOffset[iRing + 1] - panRingOffset[iRing]) +
            1;
        auto poRing = std::make_unique<OGRLinearRing>();
        poRing->setNumPoints(nPoints, /* bZeroizeNewContent = */ false);
        if (poRing->getNumPoints() < nPoints)
            return nullptr;
        poPolygon->addRingDirectly(poRing.release());
    }
    auto poFeature = std::make_unique<OGRFeature>(poDefn);
    poFeature->SetGeometryDirectly(poPolygon.release());
    poFeature->SetFID(OGRNullFID);
    if (iPixValField >= 0)
        poFeature->SetField(iPixValField, dfValue);
    return poFeature;
}

/** The points of the rings of that feature. */
void FillFeature(const xrasp::Polygons &oPolygons, int64_t iPoly,
                 OGRFeature *poFeature, const GDALGeoTransform &gt)
{
    const int64_t *panPolyOffset = oPolygons.poly_offsets.data();
    const int64_t *panRingOffset = oPolygons.ring_offsets.data();
    const int32_t *panX = oPolygons.x.data();
    const int32_t *panY = oPolygons.y.data();
    OGRPolygon *poPolygon = poFeature->GetGeometryRef()->toPolygon();
    const int64_t iFirstRing = panPolyOffset[iPoly];
    for (int64_t iRing = iFirstRing; iRing < panPolyOffset[iPoly + 1]; ++iRing)
    {
        OGRLinearRing *poRing =
            iRing == iFirstRing
                ? poPolygon->getExteriorRing()
                : poPolygon->getInteriorRing(
                      static_cast<int>(iRing - iFirstRing - 1));
        const int64_t nFirst = panRingOffset[iRing];
        const int nPoints = static_cast<int>(panRingOffset[iRing + 1] - nFirst);
        for (int i = 0; i <= nPoints; ++i)
        {
            const int64_t iVertex = nFirst + (i == nPoints ? 0 : i);
            double dfX = 0;
            double dfY = 0;
            gt.Apply(panX[iVertex], panY[iVertex], &dfX, &dfY);
            poRing->setPoint(i, dfX, dfY);
        }
    }
}

/** Make the features of all the polygons, empty, on this thread.
 *
 * The memory of a feature is taken by this thread on purpose.  Taken by all
 * the threads (tried: CORINE, 32 threads) it comes from as many arenas of
 * the allocator, which keep it when the layer is freed by its owner: the
 * peak of the process grew from 8.9 to 13.8 GB over three calls and making
 * the features slowed from 1.9 to 4.5 s, against 3.1 s here every time.
 *
 * bSkipMarker: leave out the polygons of value GP_NODATA_MARKER.  The default
 * polygonizer marks masked pixels with that value, and so takes a pixel that
 * holds it for a masked one.
 *
 * nFloatBytes: 4 or 8 when the values of the polygons are the bits of floats
 * or of doubles, 0 when they are integers.
 */
bool MakeFeatures(const xrasp::Polygons &oPolygons, OGRFeatureDefn *poDefn,
                  int iPixValField, bool bSkipMarker, int nFloatBytes,
                  std::vector<std::unique_ptr<OGRFeature>> &apoFeatures)
{
    const int64_t *panValue = oPolygons.values.data();
    const int64_t nPolygons = oPolygons.num_polygons();
    try
    {
        apoFeatures.resize(static_cast<size_t>(nPolygons));
        for (int64_t iPoly = 0; iPoly < nPolygons; ++iPoly)
        {
            double dfValue = static_cast<double>(panValue[iPoly]);
            if (nFloatBytes == 4)
            {
                const uint32_t nBits = static_cast<uint32_t>(panValue[iPoly]);
                float fValue = 0;
                memcpy(&fValue, &nBits, sizeof(fValue));
                dfValue = static_cast<double>(fValue);
            }
            else if (nFloatBytes == 8)
            {
                memcpy(&dfValue, &panValue[iPoly], sizeof(dfValue));
            }
            if (bSkipMarker && dfValue == GP_NODATA_MARKER)
                continue;
            apoFeatures[iPoly] = EmptyFeatureOf(oPolygons, iPoly, poDefn,
                                                iPixValField, dfValue);
            if (!apoFeatures[iPoly])
                return false;
        }
    }
    catch (const std::exception &)
    {
        return false;
    }
    return true;
}

/** Fill in the points of the features with all the threads, and hand the
 * features to the layer in the order panOrder gives: the layer takes them
 * as they are (a MEM layer keeps the object, without a copy).
 */
CPLErr WriteFeatures(xrasp::Polygons &oPolygons, const int32_t *panOrder,
                     std::vector<std::unique_ptr<OGRFeature>> &apoFeatures,
                     OGRLayerH hOutLayer, const GDALGeoTransform &gt,
                     int nCommitInterval, int nThreads,
                     GDALProgressFunc pfnProgress, void *pProgressArg,
                     double *pdfFillSeconds)
{
    const int64_t nPolygons = oPolygons.num_polygons();
    auto oClock = std::chrono::steady_clock::now();
    // polygons differ much in size: every thread takes the next 256
    std::atomic<int64_t> nNext{0};
    xrasp::RunParts(
        nThreads,
        [&](int)
        {
            for (;;)
            {
                const int64_t nFirst = nNext.fetch_add(256);
                if (nFirst >= nPolygons)
                    break;
                const int64_t nEnd = std::min(nFirst + 256, nPolygons);
                for (int64_t iPoly = nFirst; iPoly < nEnd; ++iPoly)
                {
                    if (apoFeatures[iPoly])
                        FillFeature(oPolygons, iPoly,
                                    apoFeatures[iPoly].get(), gt);
                }
            }
        });
    // the features hold everything now
    oPolygons = xrasp::Polygons();
    *pdfFillSeconds = Seconds(oClock);

    OGRLayer *poLayer = OGRLayer::FromHandle(hOutLayer);
    GDALDataset *poOutDS = poLayer->GetDataset();
    if (nCommitInterval != 0)
    {
        CPLErrorStateBackuper oBackuper(CPLQuietErrorHandler);
        if (!(poOutDS && poOutDS->TestCapability(ODsCTransactions) &&
              poOutDS->StartTransaction(false) == OGRERR_NONE))
        {
            nCommitInterval = 0;
        }
    }
    CPLErr eErr = CE_None;
    GIntBig nFeaturesInTransaction = 0;
    for (int64_t i = 0; i < nPolygons && eErr == CE_None; ++i)
    {
        auto &poFeature = apoFeatures[panOrder[i]];
        if (poFeature)  // not a polygon of the marker value
        {
            if (poLayer->CreateFeature(std::move(poFeature)) != OGRERR_NONE)
            {
                eErr = CE_Failure;
            }
            else if (nCommitInterval != 0)
            {
                ++nFeaturesInTransaction;
                if (nFeaturesInTransaction == nCommitInterval)
                {
                    if (poOutDS->CommitTransaction() != OGRERR_NONE ||
                        poOutDS->StartTransaction(false) != OGRERR_NONE)
                    {
                        eErr = CE_Failure;
                    }
                    nFeaturesInTransaction = 0;
                }
            }
        }
        if (eErr == CE_None && ((i + 1) % 65536 == 0 || i + 1 == nPolygons) &&
            !pfnProgress(0.6 + 0.4 * static_cast<double>(i + 1) /
                                   static_cast<double>(nPolygons),
                         "", pProgressArg))
        {
            CPLError(CE_Failure, CPLE_UserInterrupt, "User terminated");
            eErr = CE_Failure;
        }
    }
    if (nFeaturesInTransaction &&
        poOutDS->CommitTransaction() != OGRERR_NONE)
    {
        eErr = CE_Failure;
    }
    if (eErr == CE_None && nPolygons == 0)
        pfnProgress(1.0, "", pProgressArg);
    return eErr;
}

/** Whether X-Rasp polygonizes these cells as the default polygonizer does:
 * a check of cells of 64 bits and of floats, with all the threads.
 *
 * Cells of 64 bits: X-Rasp keeps one value of 64 bits for what is outside
 * the raster (xrasp::kOutside), which no cell may hold.
 *
 * Floats (bFloat; T holds their bits): GDALFPolygonize() takes two
 * neighbouring cells for equal when at most MAX_ULPS floats lie between them
 * (GDALFloatAlmostEquals).  That is not transitive: cells a, b, c in a row
 * may make one polygon while a and c, were they neighbours, would not be
 * equal -- and the default polygonizer draws no edge between two cells of one
 * polygon, whatever their values.  X-Rasp draws an edge where two neighbours
 * differ.  The two agree when every pair of neighbours is either the same
 * float or further apart than MAX_ULPS: almost equal is then the same as
 * equal bits, and X-Rasp runs on the bits.  +0 next to -0 (equal, other bits)
 * does not pass either.
 *
 * A raster that does not pass is left to the default polygonizer.
 */
template <typename T>
bool CellsAreTaken(const T *paCells, bool bFloat, int nXSize, int nYSize,
                   int nThreads)
{
    typedef typename std::make_unsigned<T>::type Bits;
    constexpr int NBITS = 8 * static_cast<int>(sizeof(T)) - 1;
    // the place of a float among the floats, as GDALFloatAlmostEquals has it
    const auto Place = [](Bits nBits)
    {
        constexpr Bits SHIFT = static_cast<Bits>(1) << NBITS;
        return static_cast<Bits>((nBits >> NBITS) != 0
                                     ? SHIFT - (nBits & ~SHIFT)
                                     : nBits + SHIFT);
    };
    const auto Apart = [&Place](T nA, T nB)
    {
        if (nA == nB)
            return true;
        const Bits nPlaceA = Place(static_cast<Bits>(nA));
        const Bits nPlaceB = Place(static_cast<Bits>(nB));
        return static_cast<Bits>(nPlaceA > nPlaceB ? nPlaceA - nPlaceB
                                                   : nPlaceB - nPlaceA) >
               static_cast<Bits>(MAX_ULPS);
    };
    const bool bWide = sizeof(T) == sizeof(int64_t);
    std::atomic<bool> bTaken{true};
    const int nParts = std::max(1, std::min(nThreads, nYSize));
    xrasp::RunParts(
        nParts,
        [&](int iPart)
        {
            const int nLastY = static_cast<int>(
                static_cast<int64_t>(nYSize) * (iPart + 1) / nParts);
            for (int iY = static_cast<int>(static_cast<int64_t>(nYSize) *
                                           iPart / nParts);
                 iY < nLastY && bTaken.load(std::memory_order_relaxed); ++iY)
            {
                const T *paRow = paCells + static_cast<size_t>(iY) * nXSize;
                const T *paBelow =
                    iY + 1 < nYSize ? paRow + nXSize : nullptr;
                bool bRow = true;
                if (bWide)
                {
                    for (int iX = 0; iX < nXSize; ++iX)
                        bRow &= static_cast<int64_t>(paRow[iX]) !=
                                xrasp::kOutside;
                }
                if (bFloat)
                {
                    for (int iX = 0; iX < nXSize; ++iX)
                    {
                        if (iX + 1 < nXSize)
                            bRow &= Apart(paRow[iX], paRow[iX + 1]);
                        if (paBelow)
                            bRow &= Apart(paRow[iX], paBelow[iX]);
                    }
                }
                if (!bRow)
                    bTaken.store(false, std::memory_order_relaxed);
            }
        });
    return bTaken.load();
}

/** The cells of the band where they already are, row after row, or null:
 * a band of a MEM dataset whose buffer has the layout X-Rasp reads. */
template <typename T>
const T *CellsInPlace(GDALRasterBand *poBand, GDALDataType eDT)
{
    if (typeid(*poBand) != typeid(MEMRasterBand))
        return nullptr;
    auto poMemBand = static_cast<MEMRasterBand *>(poBand);
    const GByte *pabyData = poMemBand->GetData();
    if (pabyData == nullptr || poMemBand->GetRasterDataType() != eDT ||
        poMemBand->GetPixelOffset() != static_cast<GSpacing>(sizeof(T)) ||
        poMemBand->GetLineOffset() !=
            static_cast<GSpacing>(sizeof(T)) * poBand->GetXSize() ||
        reinterpret_cast<uintptr_t>(pabyData) % alignof(T) != 0)
    {
        return nullptr;
    }
    // blocks written through the block cache and not flushed yet
    if (poBand->FlushCache(false) != CE_None)
        return nullptr;
    return reinterpret_cast<const T *>(pabyData);
}

/** eDT: the type the cells are read as.  bFloat: they are floats or doubles
 * (GDT_Float32, GDT_Float64), which X-Rasp takes as their bits (T is uint32_t
 * or uint64_t). */
template <typename T>
bool Run(GDALRasterBandH hSrcBand, GDALDataType eDT, bool bFloat,
         OGRLayerH hOutLayer, int iPixValField, CSLConstList papszOptions,
         GDALProgressFunc pfnProgress, void *pProgressArg, CPLErr *peErr)
{
    const int nXSize = GDALGetRasterBandXSize(hSrcBand);
    const int nYSize = GDALGetRasterBandYSize(hSrcBand);

    // The geotransform, as in GDALPolygonizeT().
    GDALGeoTransform gt;
    bool bGotGeoTransform = false;
    const char *pszDatasetForGeoRef =
        CSLFetchNameValue(papszOptions, "DATASET_FOR_GEOREF");
    if (pszDatasetForGeoRef)
    {
        auto poSrcDS = std::unique_ptr<GDALDataset>(GDALDataset::Open(
            pszDatasetForGeoRef, GDAL_OF_RASTER | GDAL_OF_VERBOSE_ERROR));
        if (poSrcDS)
            bGotGeoTransform = poSrcDS->GetGeoTransform(gt) == CE_None;
    }
    else
    {
        auto poSrcDS = GDALRasterBand::FromHandle(hSrcBand)->GetDataset();
        if (poSrcDS)
            bGotGeoTransform = poSrcDS->GetGeoTransform(gt) == CE_None;
    }
    if (!bGotGeoTransform)
        gt = GDALGeoTransform();

    // The whole band in memory: used where it is when it is there already,
    // read otherwise.  No memory for it: the default polygonizer reads line
    // by line.
    auto oStart = std::chrono::steady_clock::now();
    const T *paCells =
        CellsInPlace<T>(GDALRasterBand::FromHandle(hSrcBand), eDT);
    T *paCellsRead = nullptr;
    if (paCells == nullptr)
    {
        {
            CPLErrorStateBackuper oBackuper(CPLQuietErrorHandler);
            paCellsRead = static_cast<T *>(
                VSI_MALLOC3_VERBOSE(nXSize, nYSize, sizeof(T)));
        }
        if (paCellsRead == nullptr)
        {
            CPLDebug("GDAL", "X-Rasp: no memory for the whole band, the "
                             "default polygonizer takes over");
            return false;
        }
        if (GDALRasterIO(hSrcBand, GF_Read, 0, 0, nXSize, nYSize, paCellsRead,
                         nXSize, nYSize, eDT, 0, 0) != CE_None)
        {
            VSIFree(paCellsRead);
            *peErr = CE_Failure;
            return true;
        }
        paCells = paCellsRead;
    }
    const double dfReadSeconds = Seconds(oStart);
    const bool bCellsRead = paCellsRead != nullptr;
    if (!pfnProgress(0.1, "", pProgressArg))
    {
        VSIFree(paCellsRead);
        CPLError(CE_Failure, CPLE_UserInterrupt, "User terminated");
        *peErr = CE_Failure;
        return true;
    }

    // The polygons, the holes of each and the vertices of every ring as the
    // default polygonizer has them, the polygons by the row of their last
    // pixel.
    xrasp::Polygons oPolygons;
    xrasp::WholeProfile oProfile;
    xrasp::WholeProfile oOrderProfile;
    xrasp::WholePolygonizer<T> oPolygonizer;
    oStart = std::chrono::steady_clock::now();
    const int nThreads = ThreadCount();
    // used where GDAL does not use OpenMP
    xrasp::SetCpuPool(RunOnGDALPool);
    const int nThreadsBefore = xrasp::CpuThreads();
    xrasp::SetCpuThreads(nThreads);
    if ((bFloat || sizeof(T) == sizeof(int64_t)) &&
        !CellsAreTaken(paCells, bFloat, nXSize, nYSize, nThreads))
    {
        xrasp::SetCpuThreads(nThreadsBefore);
        VSIFree(paCellsRead);
        CPLDebug("GDAL", "X-Rasp: %s, the default polygonizer takes over",
                 bFloat ? "neighbouring cells that are almost equal and not "
                          "the same, or a cell of -0 as a double"
                        : "a cell of the smallest value of 64 bits");
        return false;
    }
    bool bDone = false;
    try
    {
        xrasp::WholeOptions oOptions;
        // corners per thread in the first step: on a CPU the largest chunk
        oOptions.corner_chunk = 255;
        // the first level of the ranking by walking, not by pointer jumping
        oOptions.rank_walk = true;
        oPolygonizer.FindPolygons(paCells, static_cast<int64_t>(nYSize),
                                  static_cast<int64_t>(nXSize), oOptions,
                                  &oPolygons, nullptr, &oProfile);
        bDone = true;
    }
    catch (const std::exception &e)
    {
        // more than 2^31 - 1 vertices, or out of memory: nothing was written
        CPLDebug("GDAL", "X-Rasp: %s; the default polygonizer takes over",
                 e.what());
    }
    xrasp::SetCpuThreads(nThreadsBefore);
    if (!bDone)
    {
        VSIFree(paCellsRead);
        return false;
    }
    const double dfPolygonizeSeconds = Seconds(oStart);
    CPLDebug("GDAL",
             "X-Rasp: " CPL_FRMT_GIB " polygons with %d threads (%s)",
             static_cast<GIntBig>(oPolygons.num_polygons()), nThreads,
             xrasp::CpuThreadsOf());
    if (!pfnProgress(0.5, "", pProgressArg))
    {
        VSIFree(paCellsRead);
        CPLError(CE_Failure, CPLE_UserInterrupt, "User terminated");
        *peErr = CE_Failure;
        return true;
    }

    // What is left to find is the order of the polygons that end in the
    // same row, and what is left to do on this thread is to make the
    // features: the two at the same time, the order on a thread of its own
    // with the other threads of the machine.
    oStart = std::chrono::steady_clock::now();
    xrasp::DArray<int32_t> oOrder;
    bool bOrdered = false;
    std::string osOrderError;
    const auto FindOrder = [&]()
    {
        // one core is this thread's (tried on fewer, 16, 8 and 4 of 32: no
        // steady gain for making the features, and the order takes longer)
        xrasp::SetCpuThreads(std::max(1, nThreads - 1));
        try
        {
            oPolygonizer.FindOrder(&oOrder, &oOrderProfile);
            bOrdered = true;
        }
        catch (const std::exception &e)
        {
            osOrderError = e.what();
        }
    };
    std::thread oOrderThread;
    try
    {
        oOrderThread = std::thread(FindOrder);
    }
    catch (const std::exception &)
    {
        // no thread to be had: one after the other
    }
    std::vector<std::unique_ptr<OGRFeature>> apoFeatures;
    const bool bMade = MakeFeatures(
        oPolygons, OGRLayer::FromHandle(hOutLayer)->GetLayerDefn(),
        iPixValField,
        /* bSkipMarker = */ eDT == GDT_Int32 || eDT == GDT_Int64 || bFloat,
        bFloat ? static_cast<int>(sizeof(T)) : 0, apoFeatures);
    const double dfMakeSeconds = Seconds(oStart);
    if (oOrderThread.joinable())
    {
        oOrderThread.join();
    }
    else
    {
        FindOrder();
        xrasp::SetCpuThreads(nThreadsBefore);
    }
    const double dfOrderAndMakeSeconds = Seconds(oStart);
    VSIFree(paCellsRead);
    if (!bOrdered || !bMade)
    {
        // out of memory: nothing was written
        CPLDebug("GDAL", "X-Rasp: %s; the default polygonizer takes over",
                 bOrdered ? "no memory for the features" : osOrderError.c_str());
        return false;
    }
    if (!pfnProgress(0.6, "", pProgressArg))
    {
        CPLError(CE_Failure, CPLE_UserInterrupt, "User terminated");
        *peErr = CE_Failure;
        return true;
    }

    oStart = std::chrono::steady_clock::now();
    double dfFillSeconds = 0;
    *peErr = WriteFeatures(
        oPolygons, oOrder.data(), apoFeatures, hOutLayer, gt,
        atoi(CSLFetchNameValueDef(papszOptions, "COMMIT_INTERVAL", "100000")),
        nThreads, pfnProgress, pProgressArg, &dfFillSeconds);
    const double dfWriteSeconds = Seconds(oStart);

    // Where the time and the memory went, for CPL_DEBUG.
    std::string osSteps;
    std::string osBytes;
    for (const auto &oStep : oProfile.steps)
    {
        osSteps += CPLSPrintf(" %s %.0f", oStep.name.c_str(), oStep.ms);
        osBytes += CPLSPrintf(" %s %.2f", oStep.name.c_str(),
                              static_cast<double>(oStep.bytes) / 1e9);
    }
    std::string osOrderSteps;
    for (const auto &oStep : oOrderProfile.steps)
    {
        osOrderSteps += CPLSPrintf(" %s %.0f", oStep.name.c_str(), oStep.ms);
        osBytes += CPLSPrintf(" %s %.2f", oStep.name.c_str(),
                              static_cast<double>(oStep.bytes) / 1e9);
    }
    CPLDebug("GDAL",
             "X-Rasp: seconds: cells %.2f (%s), polygons %.2f, order and "
             "features made %.2f, features filled %.2f, handed to the layer "
             "%.2f; steps in ms:%s; at the same time: features made %.0f, "
             "order%s",
             dfReadSeconds, bCellsRead ? "read" : "in place",
             dfPolygonizeSeconds, dfOrderAndMakeSeconds, dfFillSeconds,
             dfWriteSeconds - dfFillSeconds, osSteps.c_str(),
             dfMakeSeconds * 1000, osOrderSteps.c_str());
    CPLDebug("GDAL",
             "X-Rasp: GB: cells read %.2f, arrays at most %.2f; arrays after "
             "each step:%s",
             bCellsRead ? static_cast<double>(nXSize) * nYSize * sizeof(T) / 1e9
                        : 0.0,
             static_cast<double>(std::max(oProfile.peak_bytes,
                                          oOrderProfile.peak_bytes)) /
                 1e9,
             osBytes.c_str());
    return true;
}

}  // namespace

/** What X-Rasp asks of a request, whatever the cells. */
static bool TakesRequest(GDALRasterBandH hSrcBand, GDALRasterBandH hMaskBand,
                         OGRLayerH hOutLayer, CSLConstList papszOptions)
{
    if (hSrcBand == nullptr || hOutLayer == nullptr || hMaskBand != nullptr)
        return false;
    if (CSLFetchNameValue(papszOptions, "8CONNECTED"))
        return false;
    if (!CPLTestBool(CPLGetConfigOption("GDAL_POLYGONIZE_XRASP", "YES")))
        return false;
    if (!OGR_L_TestCapability(hOutLayer, OLCSequentialWrite))
        return false;
    return GDALGetRasterBandXSize(hSrcBand) >= 1 &&
           GDALGetRasterBandYSize(hSrcBand) >= 1;
}

bool GDALFPolygonizeXRasp(GDALRasterBandH hSrcBand, GDALRasterBandH hMaskBand,
                          OGRLayerH hOutLayer, int iPixValField,
                          CSLConstList papszOptions,
                          GDALProgressFunc pfnProgress, void *pProgressArg,
                          CPLErr *peErr)
{
    if (!TakesRequest(hSrcBand, hMaskBand, hOutLayer, papszOptions))
        return false;
    if (pfnProgress == nullptr)
        pfnProgress = GDALDummyProgress;
    // GDALFPolygonize() reads a Float64 band as doubles and a band of any
    // other type as floats.
    if (GDALGetRasterDataType(hSrcBand) == GDT_Float64)
        return Run<uint64_t>(hSrcBand, GDT_Float64, /* bFloat = */ true,
                             hOutLayer, iPixValField, papszOptions,
                             pfnProgress, pProgressArg, peErr);
    return Run<uint32_t>(hSrcBand, GDT_Float32, /* bFloat = */ true,
                         hOutLayer, iPixValField, papszOptions, pfnProgress,
                         pProgressArg, peErr);
}

bool GDALPolygonizeXRasp(GDALRasterBandH hSrcBand, GDALRasterBandH hMaskBand,
                         OGRLayerH hOutLayer, int iPixValField,
                         CSLConstList papszOptions,
                         GDALProgressFunc pfnProgress, void *pProgressArg,
                         CPLErr *peErr)
{
    if (!TakesRequest(hSrcBand, hMaskBand, hOutLayer, papszOptions))
        return false;
    if (pfnProgress == nullptr)
        pfnProgress = GDALDummyProgress;

    const GDALDataType eDT = GDALGetRasterDataType(hSrcBand);
    switch (eDT)
    {
        case GDT_Byte:
            return Run<uint8_t>(hSrcBand, eDT, false, hOutLayer, iPixValField,
                                papszOptions, pfnProgress, pProgressArg,
                                peErr);
        case GDT_Int8:
            return Run<int8_t>(hSrcBand, eDT, false, hOutLayer, iPixValField,
                               papszOptions, pfnProgress, pProgressArg, peErr);
        case GDT_UInt16:
            return Run<uint16_t>(hSrcBand, eDT, false, hOutLayer, iPixValField,
                                 papszOptions, pfnProgress, pProgressArg,
                                 peErr);
        case GDT_Int16:
            return Run<int16_t>(hSrcBand, eDT, false, hOutLayer, iPixValField,
                                papszOptions, pfnProgress, pProgressArg,
                                peErr);
        case GDT_UInt32:
            return Run<uint32_t>(hSrcBand, eDT, false, hOutLayer, iPixValField,
                                 papszOptions, pfnProgress, pProgressArg,
                                 peErr);
        case GDT_Int32:
            return Run<int32_t>(hSrcBand, eDT, false, hOutLayer, iPixValField,
                                papszOptions, pfnProgress, pProgressArg,
                                peErr);
        default:
            // Any other band is read as integers of 64 bits, as
            // GDALPolygonize() reads every band.
            return Run<int64_t>(hSrcBand, GDT_Int64, false, hOutLayer,
                                iPixValField, papszOptions, pfnProgress,
                                pProgressArg, peErr);
    }
}
