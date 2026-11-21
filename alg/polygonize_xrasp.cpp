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
#include <cerrno>
#include <climits>
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
#include "ogr_recordbatch.h"
#include "ogrsf_frmts.h"

#include "xrasp/core/polygonize_whole.h"
#include "xrasp/core/polygonizer_bands.h"
#include "xrasp/backend/threads.h"

namespace
{

/** The threads of X-Rasp where GDAL does not use OpenMP: GDAL's worker threads
 * (xrasp/backend/threads.h).  pfnPart(pJob, i) for i = 0 .. nParts-1, the
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

/** The two columns of the array while they grow: polygons are appended as
 * the bands of the raster are finished.  The WKB of all of them is one block
 * that is made larger as needed -- more of it is asked for than is written,
 * and what is not written is not memory of the process yet.
 */
struct WkbColumn
{
    GByte *pabyWkb = nullptr;
    int64_t nBytes = 0;
    int64_t nCapacity = 0;
    std::vector<int64_t> anEnd{};   // where the WKB of every polygon ends
    std::vector<double> adfValue{};
    bool bNoMemory = false;

    WkbColumn() = default;
    WkbColumn(const WkbColumn &) = delete;
    WkbColumn &operator=(const WkbColumn &) = delete;

    ~WkbColumn()
    {
        VSIFree(pabyWkb);
    }

    /** Nothing written. */
    void Reset()
    {
        VSIFree(pabyWkb);
        pabyWkb = nullptr;
        nBytes = 0;
        nCapacity = 0;
        std::vector<int64_t>().swap(anEnd);
        std::vector<double>().swap(adfValue);
        bNoMemory = false;
    }

    bool Reserve(int64_t nMore)
    {
        if (nBytes + nMore <= nCapacity)
            return true;
        const int64_t nNew = std::max<int64_t>(
            {2 * nCapacity, nBytes + nMore, int64_t{1} << 26});
        GByte *pabyNew = static_cast<GByte *>(
            VSI_REALLOC_VERBOSE(pabyWkb, static_cast<size_t>(nNew)));
        if (pabyNew == nullptr)
        {
            bNoMemory = true;
            return false;
        }
        pabyWkb = pabyNew;
        nCapacity = nNew;
        return true;
    }
};

/** What the Arrow array is written from: the polygons as X-Rasp has them,
 * their order, and how a value and a point are written.  No feature is
 * made.
 */
struct ArrowJob
{
    xrasp::Polygons oPolygons{};
    xrasp::DArray<int32_t> oOrder{};
    GDALGeoTransform gt{};
    bool bSkipMarker = false;
    int nFloatBytes = 0;
    int nThreads = 1;
    // Band after band (GDAL_POLYGONIZE_BANDS): the polygons are here and not
    // in oPolygons, their vertices in one piece per band.  The polygonizer
    // refers to the configuration and the raster beside it.
    bool bBandsAllowed = false;
    xrasp::PolygonizeConfig oBandConfig{};
    xrasp::Raster oBandRaster{};
    std::unique_ptr<xrasp::PolygonizerBands> poBands{};
    // the polygons written so far (GDAL_POLYGONIZE_BAND_WRITE), and whether
    // the bands are still to be drained
    WkbColumn oColumn{};
    bool bDrainBands = false;
    // seconds, for CPL_DEBUG: inside the polygonizer handing bands out (of that, writing their WKB)
    double dfEmitSeconds = 0;
    double dfSinkSeconds = 0;
};

/** The cell type of the band version for T, or false when it has none (cells
 * of 64 bits). */
template <typename T> bool BandCellType(xrasp::CellType *peType)
{
    if (sizeof(T) > 4)
        return false;
    if (sizeof(T) == 1)
        *peType = std::is_signed<T>::value ? xrasp::CellType::kI8
                                           : xrasp::CellType::kU8;
    else if (sizeof(T) == 2)
        *peType = std::is_signed<T>::value ? xrasp::CellType::kI16
                                           : xrasp::CellType::kU16;
    else
        *peType = std::is_signed<T>::value ? xrasp::CellType::kI32
                                           : xrasp::CellType::kU32;
    return true;
}

/** Whether a raster of this many cells is polygonized band after band:
 * GDAL_POLYGONIZE_BANDS = YES, NO or AUTO (the default: from 2^26 cells on).
 * In bands the arrays of the polygonizer are those of one band, a few million
 * vertices, where the whole raster at once needs them for all its vertices;
 * the polygons are the same, in another order (see GDALPolygonizeToArrow()).
 */
bool InBands(int nXSize, int nYSize)
{
    const char *pszBands = CPLGetConfigOption("GDAL_POLYGONIZE_BANDS", "AUTO");
    if (EQUAL(pszBands, "AUTO"))
        return static_cast<int64_t>(nXSize) * nYSize >= (int64_t{1} << 26);
    return CPLTestBool(pszBands);
}

/** The value of a polygon as the field of its feature has it (see
 * MakeFeatures()). */
inline double ValueOfPolygon(int64_t nValue, int nFloatBytes)
{
    double dfValue = static_cast<double>(nValue);
    if (nFloatBytes == 4)
    {
        const uint32_t nBits = static_cast<uint32_t>(nValue);
        float fValue = 0;
        memcpy(&fValue, &nBits, sizeof(fValue));
        dfValue = static_cast<double>(fValue);
    }
    else if (nFloatBytes == 8)
    {
        memcpy(&dfValue, &nValue, sizeof(dfValue));
    }
    return dfValue;
}

/** The bytes of a polygon as ISO WKB: 9 for the polygon, 4 for every ring,
 * 16 for every point, a ring closed with its first point. */
inline int64_t WkbSizeOfPolygon(const xrasp::Polygons &oPolygons,
                                int64_t iPoly)
{
    const int64_t *panPolyOffset = oPolygons.poly_offsets.data();
    const int64_t *panRingOffset = oPolygons.ring_offsets.data();
    const int64_t nRings = panPolyOffset[iPoly + 1] - panPolyOffset[iPoly];
    const int64_t nVertices = panRingOffset[panPolyOffset[iPoly + 1]] -
                              panRingOffset[panPolyOffset[iPoly]];
    return 9 + 4 * nRings + 16 * (nVertices + nRings);
}

/** The polygon as little-endian ISO WKB, what OGR_G_ExportToIsoWkb() writes
 * of the polygon FillFeature() makes. */
void WritePolygonAsWkb(const xrasp::Polygons &oPolygons, int64_t iPoly,
                       const GDALGeoTransform &gt, GByte *pabyOut)
{
    const int64_t *panPolyOffset = oPolygons.poly_offsets.data();
    const int64_t *panRingOffset = oPolygons.ring_offsets.data();
    const int32_t *panX = oPolygons.x.data();
    const int32_t *panY = oPolygons.y.data();
    *pabyOut++ = 1;  // wkbNDR
    const uint32_t nType = 3;  // wkbPolygon
    memcpy(pabyOut, &nType, 4);
    pabyOut += 4;
    const uint32_t nRings =
        static_cast<uint32_t>(panPolyOffset[iPoly + 1] - panPolyOffset[iPoly]);
    memcpy(pabyOut, &nRings, 4);
    pabyOut += 4;
    for (int64_t iRing = panPolyOffset[iPoly]; iRing < panPolyOffset[iPoly + 1];
         ++iRing)
    {
        const int64_t nFirst = panRingOffset[iRing];
        const int64_t nPoints = panRingOffset[iRing + 1] - nFirst;
        const uint32_t nPointsClosed = static_cast<uint32_t>(nPoints + 1);
        memcpy(pabyOut, &nPointsClosed, 4);
        pabyOut += 4;
        for (int64_t i = 0; i <= nPoints; ++i)
        {
            const int64_t iVertex = nFirst + (i == nPoints ? 0 : i);
            double adfXY[2] = {0, 0};
            gt.Apply(panX[iVertex], panY[iVertex], &adfXY[0], &adfXY[1]);
            memcpy(pabyOut, adfXY, 16);
            pabyOut += 16;
        }
    }
}

void ReleaseArrowSchema(struct ArrowSchema *psSchema)
{
    for (int64_t i = 0; i < psSchema->n_children; ++i)
    {
        if (psSchema->children[i]->release)
            psSchema->children[i]->release(psSchema->children[i]);
        CPLFree(psSchema->children[i]);
    }
    CPLFree(psSchema->children);
    psSchema->release = nullptr;
}

/** struct<geom: binary not null, value: double not null>; bLarge: geom is a
 * large binary (offsets of 64 bits). */
void InitArrowSchema(struct ArrowSchema *psOut, bool bLarge)
{
    memset(psOut, 0, sizeof(*psOut));
    psOut->format = "+s";
    psOut->name = "";
    psOut->release = ReleaseArrowSchema;
    psOut->n_children = 2;
    psOut->children = static_cast<struct ArrowSchema **>(
        CPLCalloc(2, sizeof(struct ArrowSchema *)));
    const char *const apszFormat[2] = {bLarge ? "Z" : "z", "g"};
    const char *const apszName[2] = {"geom", "value"};
    for (int i = 0; i < 2; ++i)
    {
        psOut->children[i] = static_cast<struct ArrowSchema *>(
            CPLCalloc(1, sizeof(struct ArrowSchema)));
        psOut->children[i]->format = apszFormat[i];
        psOut->children[i]->name = apszName[i];
        psOut->children[i]->release = ReleaseArrowSchema;
    }
}

void ReleaseArrowArray(struct ArrowArray *psArray)
{
    for (int64_t i = 0; i < psArray->n_children; ++i)
    {
        if (psArray->children[i]->release)
            psArray->children[i]->release(psArray->children[i]);
        CPLFree(psArray->children[i]);
    }
    CPLFree(psArray->children);
    for (int64_t i = 0; i < psArray->n_buffers; ++i)
        VSIFree(const_cast<void *>(psArray->buffers[i]));
    CPLFree(psArray->buffers);
    psArray->release = nullptr;
}

/** An array of nLength values without nulls and with nBuffers buffers, the
 * first of which (the validity) is not there. */
void InitArrowArray(struct ArrowArray *psArray, int64_t nLength, int nBuffers)
{
    memset(psArray, 0, sizeof(*psArray));
    psArray->length = nLength;
    psArray->n_buffers = nBuffers;
    psArray->buffers =
        static_cast<const void **>(CPLCalloc(nBuffers, sizeof(void *)));
    psArray->release = ReleaseArrowArray;
}

/** All the polygons as one array, in the order of the default polygonizer:
 * where the WKB of every polygon starts is known before any is written, so
 * one block is taken for all of them and all the threads write into it.
 *
 * Returns 0, ENOMEM, or EOVERFLOW when the WKB of all the polygons is more
 * than the 2 GB the offsets of a binary array can say.
 */
int WriteArrowArray(const ArrowJob &oJob, struct ArrowArray *psOut)
{
    const xrasp::Polygons &oPolygons = oJob.oPolygons;
    const int64_t nPolygons = oPolygons.num_polygons();
    const int64_t *panValue = oPolygons.values.data();
    const int32_t *panOrder = oJob.oOrder.data();

    // the polygons that are written (not those of the marker value) ...
    std::vector<int32_t> anPoly;
    int64_t nBytes = 0;
    try
    {
        anPoly.reserve(static_cast<size_t>(nPolygons));
        for (int64_t i = 0; i < nPolygons; ++i)
        {
            const int32_t iPoly = panOrder[i];
            if (oJob.bSkipMarker &&
                ValueOfPolygon(panValue[iPoly], oJob.nFloatBytes) ==
                    GP_NODATA_MARKER)
            {
                continue;
            }
            nBytes += WkbSizeOfPolygon(oPolygons, iPoly);
            if (nBytes > INT32_MAX)
                return EOVERFLOW;
            anPoly.push_back(iPoly);
        }
    }
    catch (const std::exception &)
    {
        return ENOMEM;
    }
    const int64_t nCount = static_cast<int64_t>(anPoly.size());

    int32_t *panOffset = static_cast<int32_t *>(
        VSI_MALLOC2_VERBOSE(static_cast<size_t>(nCount) + 1, 4));
    GByte *pabyWkb = static_cast<GByte *>(
        VSI_MALLOC_VERBOSE(static_cast<size_t>(std::max<int64_t>(nBytes, 1))));
    double *padfValue = static_cast<double *>(VSI_MALLOC2_VERBOSE(
        static_cast<size_t>(std::max<int64_t>(nCount, 1)), 8));
    if (panOffset == nullptr || pabyWkb == nullptr || padfValue == nullptr)
    {
        VSIFree(panOffset);
        VSIFree(pabyWkb);
        VSIFree(padfValue);
        return ENOMEM;
    }
    // ... and where each of them starts
    panOffset[0] = 0;
    for (int64_t i = 0; i < nCount; ++i)
    {
        panOffset[i + 1] = panOffset[i] + static_cast<int32_t>(
                                              WkbSizeOfPolygon(oPolygons, anPoly[i]));
    }

    // polygons differ much in size: every thread takes the next 256
    std::atomic<int64_t> nNext{0};
    xrasp::SetCpuPool(RunOnGDALPool);
    xrasp::RunParts(
        oJob.nThreads,
        [&](int)
        {
            for (;;)
            {
                const int64_t nFirst = nNext.fetch_add(256);
                if (nFirst >= nCount)
                    break;
                const int64_t nEnd = std::min(nFirst + 256, nCount);
                for (int64_t i = nFirst; i < nEnd; ++i)
                {
                    WritePolygonAsWkb(oPolygons, anPoly[i], oJob.gt,
                                      pabyWkb + panOffset[i]);
                    padfValue[i] =
                        ValueOfPolygon(panValue[anPoly[i]], oJob.nFloatBytes);
                }
            }
        });

    InitArrowArray(psOut, nCount, 1);
    psOut->n_children = 2;
    psOut->children = static_cast<struct ArrowArray **>(
        CPLCalloc(2, sizeof(struct ArrowArray *)));
    psOut->children[0] =
        static_cast<struct ArrowArray *>(CPLCalloc(1, sizeof(struct ArrowArray)));
    InitArrowArray(psOut->children[0], nCount, 3);
    psOut->children[0]->buffers[1] = panOffset;
    psOut->children[0]->buffers[2] = pabyWkb;
    psOut->children[1] =
        static_cast<struct ArrowArray *>(CPLCalloc(1, sizeof(struct ArrowArray)));
    InitArrowArray(psOut->children[1], nCount, 2);
    psOut->children[1]->buffers[1] = padfValue;
    return 0;
}

/** Polygons appended to the columns.  oSource says what they are:
 *
 *   Count()            the candidates
 *   Complete(i)        whether candidate i is a polygon (all its rings known)
 *   Value(i)           its value, as the polygonizer has it
 *   FirstRing(i), EndRing(i)   its rings, the outer one first
 *   Ring(k, &x, &y, &n)        the n vertices of ring k (not closed)
 *
 * The bytes of every polygon are counted first, with all the threads; then
 * all the threads write, each polygon at its place.  False when memory ran
 * out (poColumn->bNoMemory).
 */
template <typename Source>
bool AppendPolygons(const ArrowJob &oJob, const Source &oSource,
                    WkbColumn *poColumn)
{
    const int64_t nCandidates = oSource.Count();
    const GDALGeoTransform &gt = oJob.gt;
    std::vector<int64_t> anPoly;
    std::vector<int64_t> anStart;
    try
    {
        anPoly.reserve(static_cast<size_t>(nCandidates));
        for (int64_t i = 0; i < nCandidates; ++i)
        {
            if (!oSource.Complete(i))
                continue;
            // the default polygonizer takes a pixel of the marker value for
            // a masked one
            if (oJob.bSkipMarker &&
                ValueOfPolygon(oSource.Value(i), oJob.nFloatBytes) ==
                    GP_NODATA_MARKER)
            {
                continue;
            }
            anPoly.push_back(i);
        }
        anStart.resize(anPoly.size() + 1);
        poColumn->anEnd.reserve(poColumn->anEnd.size() + anPoly.size());
        poColumn->adfValue.reserve(poColumn->adfValue.size() + anPoly.size());
    }
    catch (const std::exception &)
    {
        poColumn->bNoMemory = true;
        return false;
    }
    const int64_t nCount = static_cast<int64_t>(anPoly.size());
    if (nCount == 0)
        return true;

    xrasp::SetCpuPool(RunOnGDALPool);
    {
        std::atomic<int64_t> nNext{0};
        xrasp::RunParts(
            oJob.nThreads,
            [&](int)
            {
                for (;;)
                {
                    const int64_t nFirst = nNext.fetch_add(4096);
                    if (nFirst >= nCount)
                        break;
                    const int64_t nEnd = std::min(nFirst + 4096, nCount);
                    for (int64_t i = nFirst; i < nEnd; ++i)
                    {
                        const int64_t iPoly = anPoly[i];
                        int64_t nSize = 9;
                        for (int64_t k = oSource.FirstRing(iPoly);
                             k < oSource.EndRing(iPoly); ++k)
                        {
                            const int32_t *panX = nullptr;
                            const int32_t *panY = nullptr;
                            int64_t nPoints = 0;
                            oSource.Ring(k, &panX, &panY, &nPoints);
                            nSize += 4 + 16 * (nPoints + 1);
                        }
                        anStart[i + 1] = nSize;
                    }
                }
            });
    }
    anStart[0] = 0;
    for (int64_t i = 0; i < nCount; ++i)
        anStart[i + 1] += anStart[i];
    if (!poColumn->Reserve(anStart[nCount]))
        return false;
    GByte *const pabyBase = poColumn->pabyWkb + poColumn->nBytes;
    {
        std::atomic<int64_t> nNext{0};
        xrasp::RunParts(
            oJob.nThreads,
            [&](int)
            {
                for (;;)
                {
                    const int64_t nFirst = nNext.fetch_add(256);
                    if (nFirst >= nCount)
                        break;
                    const int64_t nEnd = std::min(nFirst + 256, nCount);
                    for (int64_t i = nFirst; i < nEnd; ++i)
                    {
                        const int64_t iPoly = anPoly[i];
                        GByte *pabyOut = pabyBase + anStart[i];
                        *pabyOut++ = 1;  // wkbNDR
                        const uint32_t nType = 3;  // wkbPolygon
                        memcpy(pabyOut, &nType, 4);
                        pabyOut += 4;
                        const uint32_t nRings = static_cast<uint32_t>(
                            oSource.EndRing(iPoly) - oSource.FirstRing(iPoly));
                        memcpy(pabyOut, &nRings, 4);
                        pabyOut += 4;
                        for (int64_t k = oSource.FirstRing(iPoly);
                             k < oSource.EndRing(iPoly); ++k)
                        {
                            const int32_t *panX = nullptr;
                            const int32_t *panY = nullptr;
                            int64_t nPoints = 0;
                            oSource.Ring(k, &panX, &panY, &nPoints);
                            const uint32_t nPointsClosed =
                                static_cast<uint32_t>(nPoints + 1);
                            memcpy(pabyOut, &nPointsClosed, 4);
                            pabyOut += 4;
                            for (int64_t j = 0; j <= nPoints; ++j)
                            {
                                const int64_t iVertex = j == nPoints ? 0 : j;
                                double adfXY[2] = {0, 0};
                                gt.Apply(panX[iVertex], panY[iVertex],
                                         &adfXY[0], &adfXY[1]);
                                memcpy(pabyOut, adfXY, 16);
                                pabyOut += 16;
                            }
                        }
                    }
                }
            });
    }
    for (int64_t i = 0; i < nCount; ++i)
    {
        poColumn->anEnd.push_back(poColumn->nBytes + anStart[i + 1]);
        poColumn->adfValue.push_back(
            ValueOfPolygon(oSource.Value(anPoly[i]), oJob.nFloatBytes));
    }
    poColumn->nBytes += anStart[nCount];
    return true;
}

/** The complete polygons of one band, as the polygonizer hands them out when
 * the band is done (xrasp::BandPolygons). */
struct PolygonsOfBand
{
    const xrasp::BandPolygons &oBand;

    int64_t Count() const
    {
        return oBand.groups;
    }
    bool Complete(int64_t i) const
    {
        return !oBand.is_hole[oBand.order[oBand.group_start[i]]];
    }
    int64_t Value(int64_t i) const
    {
        return oBand.value[oBand.order[oBand.group_start[i]]];
    }
    int64_t FirstRing(int64_t i) const
    {
        return oBand.group_start[i];
    }
    int64_t EndRing(int64_t i) const
    {
        return oBand.group_start[i + 1];
    }
    void Ring(int64_t k, const int32_t **ppanX, const int32_t **ppanY,
              int64_t *pnPoints) const
    {
        const int64_t iRing = oBand.order[k];
        const int64_t nFirst = oBand.ring_offsets[iRing];
        *ppanX = oBand.x + nFirst;
        *ppanY = oBand.y + nFirst;
        *pnPoints = oBand.ring_offsets[iRing + 1] - nFirst;
    }
};

/** The polygons the polygonizer holds when all the bands are done: all of
 * them, or, when the bands were written one by one, those whose outline
 * crosses bands.  The vertices of a ring are in the piece that holds it. */
struct PolygonsOfResult
{
    std::vector<xrasp::PolygonizerBands::Piece> aoPieces;
    const int64_t *panValue;
    const int64_t *panPolyOffset;
    const int64_t *panRingIndex;
    const int64_t *panRingOffset;
    int64_t nPolygons;

    explicit PolygonsOfResult(const xrasp::PolygonizerBands &oBands)
        : aoPieces(oBands.pieces()), panValue(oBands.result_values().data()),
          panPolyOffset(oBands.result_poly_offsets().data()),
          panRingIndex(oBands.result_ring_index().data()),
          panRingOffset(oBands.result_ring_offsets().data()),
          nPolygons(static_cast<int64_t>(oBands.result_values().size()))
    {
    }
    int64_t Count() const
    {
        return nPolygons;
    }
    bool Complete(int64_t) const
    {
        return true;
    }
    int64_t Value(int64_t i) const
    {
        return panValue[i];
    }
    int64_t FirstRing(int64_t i) const
    {
        return panPolyOffset[i];
    }
    int64_t EndRing(int64_t i) const
    {
        return panPolyOffset[i + 1];
    }
    void Ring(int64_t k, const int32_t **ppanX, const int32_t **ppanY,
              int64_t *pnPoints) const
    {
        const int64_t iRing = panRingIndex[k];
        // the piece of the ring: the last that starts at or before it
        int64_t iLow = 0;
        int64_t iHigh = static_cast<int64_t>(aoPieces.size()) - 1;
        while (iLow < iHigh)
        {
            const int64_t iMid = (iLow + iHigh + 1) / 2;
            if (aoPieces[iMid].first_ring <= iRing)
                iLow = iMid;
            else
                iHigh = iMid - 1;
        }
        const auto &oPiece = aoPieces[iLow];
        // a band that is still whole says where each of these rings starts
        // in it; otherwise the rings of the piece follow each other
        const int64_t nFirst =
            oPiece.ring_starts
                ? oPiece.ring_starts[iRing - oPiece.first_ring]
                : panRingOffset[iRing] - panRingOffset[oPiece.first_ring];
        *ppanX = oPiece.x + nFirst;
        *ppanY = oPiece.y + nFirst;
        *pnPoints = panRingOffset[iRing + 1] - panRingOffset[iRing];
    }
};

/** The polygons found band after band as one array.  The columns are given
 * to the array.
 *
 * GDAL_POLYGONIZE_BAND_WRITE says when the polygons become WKB:
 *
 *   AFTER (the default)  when the raster is polygonized: first the polygons
 *       whose outline crosses bands, then the bands one by one, each given
 *       back as soon as it is written.  While the raster is polygonized
 *       there is no WKB, and while the WKB is written there are no arrays
 *       of the polygonizer: the vertices of a band are 8 bytes a point, its
 *       WKB 16.
 *   YES   as every band is done (then the polygons that cross bands).
 *   NO    all at the end, and nothing is given back before all is written.
 *
 * *pbLarge: the WKB of all the polygons is more than 2 GB, and the offsets of
 * the array are of 64 bits (a large binary array).
 *
 * Returns 0 or ENOMEM.
 */
int WriteArrowArrayOfBands(ArrowJob &oJob, struct ArrowArray *psOut,
                           bool *pbLarge)
{
    WkbColumn &oColumn = oJob.oColumn;
    {
        // what the polygonizer still holds: the rings of the polygons whose
        // outline crosses bands
        int64_t nHeldVertices = 0;
        int64_t nHeldRings = 0;
        for (const auto &oPiece : oJob.poBands->pieces())
        {
            nHeldVertices += oPiece.vertices;
            nHeldRings += oPiece.rings;
        }
        CPLDebug("GDAL",
                 "X-Rasp: before the polygons that cross bands are written: "
                 CPL_FRMT_GIB " polygons and %.2f GB of WKB written; held: "
                 CPL_FRMT_GIB " polygons, " CPL_FRMT_GIB " rings, "
                 CPL_FRMT_GIB " vertices (%.2f GB)",
                 static_cast<GIntBig>(oColumn.anEnd.size()),
                 static_cast<double>(oColumn.nBytes) / 1e9,
                 static_cast<GIntBig>(oJob.poBands->result_values().size()),
                 static_cast<GIntBig>(nHeldRings),
                 static_cast<GIntBig>(nHeldVertices),
                 static_cast<double>(nHeldVertices) * 8 / 1e9);
    }
    const auto oCrossStart = std::chrono::steady_clock::now();
    if (oColumn.bNoMemory ||
        !AppendPolygons(oJob, PolygonsOfResult(*oJob.poBands), &oColumn))
    {
        return ENOMEM;
    }
    const double dfCrossSeconds = Seconds(oCrossStart);
    const auto oDrainStart = std::chrono::steady_clock::now();
    if (oJob.bDrainBands)
    {
        // the sink set in FindPolygonsInBands() appends band after band
        oJob.poBands->DrainBands();
        if (oColumn.bNoMemory)
            return ENOMEM;
    }
    const double dfDrainSeconds = Seconds(oDrainStart);
    // the polygonizer has nothing more to give
    oJob.poBands.reset();
    const auto oFinishStart = std::chrono::steady_clock::now();
    const int64_t nCount = static_cast<int64_t>(oColumn.anEnd.size());
    const int64_t nBytes = oColumn.nBytes;
    *pbLarge = nBytes > INT32_MAX;

    const size_t nOffsetSize = *pbLarge ? 8 : 4;
    void *pOffsets =
        VSI_MALLOC2_VERBOSE(static_cast<size_t>(nCount) + 1, nOffsetSize);
    double *padfValue = static_cast<double *>(VSI_MALLOC2_VERBOSE(
        static_cast<size_t>(std::max<int64_t>(nCount, 1)), 8));
    if (pOffsets == nullptr || padfValue == nullptr)
    {
        VSIFree(pOffsets);
        VSIFree(padfValue);
        return ENOMEM;
    }
    if (*pbLarge)
    {
        int64_t *panOffset = static_cast<int64_t *>(pOffsets);
        panOffset[0] = 0;
        for (int64_t i = 0; i < nCount; ++i)
            panOffset[i + 1] = oColumn.anEnd[i];
    }
    else
    {
        int32_t *panOffset = static_cast<int32_t *>(pOffsets);
        panOffset[0] = 0;
        for (int64_t i = 0; i < nCount; ++i)
            panOffset[i + 1] = static_cast<int32_t>(oColumn.anEnd[i]);
    }
    if (nCount > 0)
        memcpy(padfValue, oColumn.adfValue.data(),
               static_cast<size_t>(nCount) * 8);
    std::vector<int64_t>().swap(oColumn.anEnd);
    std::vector<double>().swap(oColumn.adfValue);
    // what was asked for beyond what was written goes back
    GByte *pabyWkb = static_cast<GByte *>(VSIRealloc(
        oColumn.pabyWkb, static_cast<size_t>(std::max<int64_t>(nBytes, 1))));
    if (pabyWkb == nullptr)
    {
        VSIFree(pOffsets);
        VSIFree(padfValue);
        return ENOMEM;
    }
    oColumn.pabyWkb = nullptr;
    oColumn.nBytes = 0;
    oColumn.nCapacity = 0;

    InitArrowArray(psOut, nCount, 1);
    psOut->n_children = 2;
    psOut->children = static_cast<struct ArrowArray **>(
        CPLCalloc(2, sizeof(struct ArrowArray *)));
    psOut->children[0] =
        static_cast<struct ArrowArray *>(CPLCalloc(1, sizeof(struct ArrowArray)));
    InitArrowArray(psOut->children[0], nCount, 3);
    CPLDebug("GDAL",
             "X-Rasp: bands: seconds after the polygonizer: WKB of the polygons "
             "that cross bands %.2f, draining the bands %.2f, the columns of "
             "the array %.2f",
             dfCrossSeconds, dfDrainSeconds, Seconds(oFinishStart));
    psOut->children[0]->buffers[1] = pOffsets;
    psOut->children[0]->buffers[2] = pabyWkb;
    psOut->children[1] =
        static_cast<struct ArrowArray *>(CPLCalloc(1, sizeof(struct ArrowArray)));
    InitArrowArray(psOut->children[1], nCount, 2);
    psOut->children[1]->buffers[1] = padfValue;
    return 0;
}

/** The polygons of the cells band after band, left in poJob->poBands.
 * Throws what the polygonizer throws. */
template <typename T>
void FindPolygonsInBands(const T *paCells, int nXSize, int nYSize,
                         xrasp::CellType eType, ArrowJob *poJob)
{
    xrasp::PolygonizeConfig &oConfig = poJob->oBandConfig;
    oConfig.mode = "bands";
    oConfig.band_vertices = static_cast<uint64_t>(CPLAtoGIntBig(
        CPLGetConfigOption("GDAL_POLYGONIZE_BAND_VERTICES", "2000000")));
    oConfig.strip_cells = static_cast<uint64_t>(CPLAtoGIntBig(
        CPLGetConfigOption("GDAL_POLYGONIZE_STRIP_CELLS", "33554432")));
    oConfig.corner_chunk = 255;
    oConfig.link_index = true;
    oConfig.tile = 256;
    oConfig.rank_walk = true;
    oConfig.open_walk = true;
    // everything stays where the polygonizer made it: nothing is copied
    oConfig.band_on_device = true;
    xrasp::Raster &oRaster = poJob->oBandRaster;
    oRaster.h = nYSize;
    oRaster.w = nXSize;
    oRaster.type = eType;
    oRaster.cell_size = sizeof(T);
    oRaster.UseCells(paCells);
    auto poBands =
        std::make_unique<xrasp::PolygonizerBands>(oConfig, oRaster);
    // When the polygons become WKB: see WriteArrowArrayOfBands().
    const char *pszWrite =
        CPLGetConfigOption("GDAL_POLYGONIZE_BAND_WRITE", "AFTER");
    const bool bAfter = EQUAL(pszWrite, "AFTER");
    poJob->bDrainBands = false;
    if (bAfter || CPLTestBool(pszWrite))
    {
        poBands->SetBandSink(
            [poJob](const xrasp::BandPolygons &oBand)
            {
                if (!poJob->oColumn.bNoMemory)
                    AppendPolygons(*poJob, PolygonsOfBand{oBand},
                                   &poJob->oColumn);
            },
            /* at_end = */ bAfter);
        poJob->bDrainBands = bAfter;
    }
    const auto oBandsStart = std::chrono::steady_clock::now();
    while (poBands->HasStrips())
    {
        poBands->FindVerticesOfNextStrip();
        if (poBands->BandIsComplete())
        {
            poBands->SortBand();
            poBands->SplitBand();
            poBands->CloseBandRings();
            poBands->BandToHost();
        }
    }
    const auto oBandsDone = std::chrono::steady_clock::now();
    poBands->CloseOpenRings();
    const double dfOpenSeconds = Seconds(oBandsDone);
    const auto oOpenDone = std::chrono::steady_clock::now();
    poBands->MergeRings();
    if (poJob->oColumn.bNoMemory)
        throw std::bad_alloc();
    poJob->dfEmitSeconds = poBands->emit_ms() / 1000;
    poJob->dfSinkSeconds = poBands->sink_ms() / 1000;
    CPLDebug("GDAL",
             "X-Rasp: bands: seconds: the bands %.2f (of that handing their "
             "polygons out %.2f, of that writing WKB %.2f), the rings that "
             "cross bands %.2f, merging %.2f",
             std::chrono::duration<double>(oBandsDone - oBandsStart).count(),
             poJob->dfEmitSeconds, poJob->dfSinkSeconds, dfOpenSeconds,
             Seconds(oOpenDone));
    poJob->poBands = std::move(poBands);
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
 * or uint64_t).  poArrowJob: when given, no feature is written to hOutLayer
 * (which is not looked at): the polygons and their order are left there, for
 * an Arrow array to be written from. */
template <typename T>
bool Run(GDALRasterBandH hSrcBand, GDALDataType eDT, bool bFloat,
         OGRLayerH hOutLayer, int iPixValField, CSLConstList papszOptions,
         GDALProgressFunc pfnProgress, void *pProgressArg, CPLErr *peErr,
         ArrowJob *poArrowJob = nullptr)
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
    xrasp::CellType eBandType = xrasp::CellType::kU8;
    if (poArrowJob != nullptr && poArrowJob->bBandsAllowed &&
        BandCellType<T>(&eBandType) && InBands(nXSize, nYSize))
    {
        std::string osBandError;
        // what the bands read when they write their polygons
        poArrowJob->gt = gt;
        poArrowJob->bSkipMarker =
            eDT == GDT_Int32 || eDT == GDT_Int64 || bFloat;
        poArrowJob->nFloatBytes = bFloat ? static_cast<int>(sizeof(T)) : 0;
        poArrowJob->nThreads = nThreads;
        try
        {
            FindPolygonsInBands(paCells, nXSize, nYSize, eBandType,
                                poArrowJob);
        }
        catch (const std::exception &e)
        {
            // the whole raster at once is tried next
            poArrowJob->poBands.reset();
            poArrowJob->oColumn.Reset();
            osBandError = e.what();
            if (osBandError.empty())
                osBandError = "failed";
        }
        if (osBandError.empty())
        {
            xrasp::SetCpuThreads(nThreadsBefore);
            // the vertices are in the pieces, the values read: the cells
            // are not needed any more
            VSIFree(paCellsRead);
            CPLDebug("GDAL",
                     "X-Rasp: " CPL_FRMT_GIB " polygons in bands with %d "
                     "threads (%s); seconds: cells %.2f (%s), polygons %.2f",
                     static_cast<GIntBig>(
                         poArrowJob->oColumn.anEnd.size() +
                         poArrowJob->poBands->result_values().size()),
                     nThreads, xrasp::CpuThreadsOf(), dfReadSeconds,
                     bCellsRead ? "read" : "in place", Seconds(oStart));
            *peErr = CE_None;
            return true;
        }
        CPLDebug("GDAL", "X-Rasp: bands: %s; the whole raster at once",
                 osBandError.c_str());
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

    if (poArrowJob != nullptr)
    {
        // The order of the polygons that end in the same row, with all the
        // threads: there are no features to make meanwhile.
        oStart = std::chrono::steady_clock::now();
        xrasp::DArray<int32_t> oArrowOrder;
        std::string osArrowError;
        xrasp::SetCpuThreads(nThreads);
        try
        {
            oPolygonizer.FindOrder(&oArrowOrder, &oOrderProfile);
        }
        catch (const std::exception &e)
        {
            osArrowError = e.what();
            if (osArrowError.empty())
                osArrowError = "failed";
        }
        xrasp::SetCpuThreads(nThreadsBefore);
        VSIFree(paCellsRead);
        if (!osArrowError.empty())
        {
            CPLDebug("GDAL", "X-Rasp: %s; nothing was written",
                     osArrowError.c_str());
            return false;
        }
        CPLDebug("GDAL",
                 "X-Rasp: seconds: cells %.2f (%s), polygons %.2f, order "
                 "%.2f",
                 dfReadSeconds, bCellsRead ? "read" : "in place",
                 dfPolygonizeSeconds, Seconds(oStart));
        poArrowJob->oPolygons = std::move(oPolygons);
        poArrowJob->oOrder = std::move(oArrowOrder);
        poArrowJob->gt = gt;
        poArrowJob->bSkipMarker =
            eDT == GDT_Int32 || eDT == GDT_Int64 || bFloat;
        poArrowJob->nFloatBytes = bFloat ? static_cast<int>(sizeof(T)) : 0;
        poArrowJob->nThreads = nThreads;
        *peErr = CE_None;
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

/** What X-Rasp asks of a request, whatever the cells and whatever is written. */
static bool TakesBand(GDALRasterBandH hSrcBand, GDALRasterBandH hMaskBand,
                      CSLConstList papszOptions)
{
    if (hSrcBand == nullptr || hMaskBand != nullptr)
        return false;
    if (CSLFetchNameValue(papszOptions, "8CONNECTED"))
        return false;
    if (!CPLTestBool(CPLGetConfigOption("GDAL_POLYGONIZE_XRASP", "YES")))
        return false;
    return GDALGetRasterBandXSize(hSrcBand) >= 1 &&
           GDALGetRasterBandYSize(hSrcBand) >= 1;
}

/** What X-Rasp asks of a request that writes features. */
static bool TakesRequest(GDALRasterBandH hSrcBand, GDALRasterBandH hMaskBand,
                         OGRLayerH hOutLayer, CSLConstList papszOptions)
{
    return hOutLayer != nullptr &&
           TakesBand(hSrcBand, hMaskBand, papszOptions) &&
           OGR_L_TestCapability(hOutLayer, OLCSequentialWrite);
}

/** The cells as GDALFPolygonize() reads them: a Float64 band as doubles and a
 * band of any other type as floats. */
static bool RunFloat(GDALRasterBandH hSrcBand, OGRLayerH hOutLayer,
                     int iPixValField, CSLConstList papszOptions,
                     GDALProgressFunc pfnProgress, void *pProgressArg,
                     CPLErr *peErr, ArrowJob *poArrowJob)
{
    if (GDALGetRasterDataType(hSrcBand) == GDT_Float64)
        return Run<uint64_t>(hSrcBand, GDT_Float64, /* bFloat = */ true,
                             hOutLayer, iPixValField, papszOptions,
                             pfnProgress, pProgressArg, peErr, poArrowJob);
    return Run<uint32_t>(hSrcBand, GDT_Float32, /* bFloat = */ true,
                         hOutLayer, iPixValField, papszOptions, pfnProgress,
                         pProgressArg, peErr, poArrowJob);
}

/** The cells as GDALPolygonize() reads them: integers. */
static bool RunInteger(GDALRasterBandH hSrcBand, OGRLayerH hOutLayer,
                       int iPixValField, CSLConstList papszOptions,
                       GDALProgressFunc pfnProgress, void *pProgressArg,
                       CPLErr *peErr, ArrowJob *poArrowJob)
{
    const GDALDataType eDT = GDALGetRasterDataType(hSrcBand);
    switch (eDT)
    {
        case GDT_Byte:
            return Run<uint8_t>(hSrcBand, eDT, false, hOutLayer, iPixValField,
                                papszOptions, pfnProgress, pProgressArg,
                                peErr, poArrowJob);
        case GDT_Int8:
            return Run<int8_t>(hSrcBand, eDT, false, hOutLayer, iPixValField,
                               papszOptions, pfnProgress, pProgressArg, peErr,
                               poArrowJob);
        case GDT_UInt16:
            return Run<uint16_t>(hSrcBand, eDT, false, hOutLayer, iPixValField,
                                 papszOptions, pfnProgress, pProgressArg,
                                 peErr, poArrowJob);
        case GDT_Int16:
            return Run<int16_t>(hSrcBand, eDT, false, hOutLayer, iPixValField,
                                papszOptions, pfnProgress, pProgressArg,
                                peErr, poArrowJob);
        case GDT_UInt32:
            return Run<uint32_t>(hSrcBand, eDT, false, hOutLayer, iPixValField,
                                 papszOptions, pfnProgress, pProgressArg,
                                 peErr, poArrowJob);
        case GDT_Int32:
            return Run<int32_t>(hSrcBand, eDT, false, hOutLayer, iPixValField,
                                papszOptions, pfnProgress, pProgressArg,
                                peErr, poArrowJob);
        default:
            // Any other band is read as integers of 64 bits, as
            // GDALPolygonize() reads every band.
            return Run<int64_t>(hSrcBand, GDT_Int64, false, hOutLayer,
                                iPixValField, papszOptions, pfnProgress,
                                pProgressArg, peErr, poArrowJob);
    }
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
    return RunFloat(hSrcBand, hOutLayer, iPixValField, papszOptions,
                    pfnProgress, pProgressArg, peErr, nullptr);
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
    return RunInteger(hSrcBand, hOutLayer, iPixValField, papszOptions,
                      pfnProgress, pProgressArg, peErr, nullptr);
}

/************************************************************************/
/*                       GDALPolygonizeToArrow()                        */
/************************************************************************/

/**
 * \brief Polygonize a raster band into one Arrow array.
 *
 * The polygons GDALPolygonize() (bFloat = FALSE) or GDALFPolygonize()
 * (bFloat = TRUE) writes to a layer of polygons with one field of type
 * OFTReal, in the same order, as one struct array of two columns without
 * nulls: "geom" (binary), the polygon as little-endian ISO WKB, and "value"
 * (double), the value of its pixels.  No feature and no layer is made: the
 * WKB of all the polygons is one block of memory, written by all the
 * threads from the arrays of the polygonizer.
 *
 * A large raster is polygonized band after band (configuration option
 * GDAL_POLYGONIZE_BANDS = YES, NO or AUTO, the default: from 2^26 cells on;
 * not for cells of 64 bits).  The arrays of the polygonizer are then those of
 * one band, not of the whole raster.  The polygons are the same, but not in
 * the order of GDALPolygonize(): they come band after band, the holes of a
 * polygon in the order they were found, and a ring starts at another of its
 * vertices.  When their WKB is more than 2 GB, "geom" is a large binary
 * array (offsets of 64 bits): the caller reads the format of the schema.
 * Without bands the order is that of GDALPolygonize() and results of more
 * than 2 GB of WKB are not answered.
 *
 * Only the requests X-Rasp takes are answered (no mask band, 4
 * connectedness, cells without the cases listed in alg/xrasp/README.md).
 * For any other, and with the configuration option
 * GDAL_POLYGONIZE_ARROW=NO, CE_Failure is returned with CPLE_NotSupported
 * and the two out parameters are left untouched: the caller polygonizes
 * into a layer.
 *
 * Options: 8CONNECTED (not supported here), DATASET_FOR_GEOREF as for
 * GDALPolygonize().
 *
 * @param hSrcBand the band to polygonize
 * @param hMaskBand must be NULL
 * @param bFloat whether the cells are read as GDALFPolygonize() reads them
 * @param papszOptions see above
 * @param pfnProgress callback for reporting progress or NULL
 * @param pProgressArg argument of pfnProgress
 * @param out_schema the type of the array, to be released by the caller
 * with out_schema->release(out_schema) when CE_None is returned
 * @param out_array the array, to be released by the caller with
 * out_array->release(out_array) when CE_None is returned
 * @return CE_None on success or CE_Failure
 */
CPLErr GDALPolygonizeToArrow(GDALRasterBandH hSrcBand,
                             GDALRasterBandH hMaskBand, int bFloat,
                             char **papszOptions, GDALProgressFunc pfnProgress,
                             void *pProgressArg,
                             struct ArrowSchema *out_schema,
                             struct ArrowArray *out_array)
{
    VALIDATE_POINTER1(hSrcBand, "GDALPolygonizeToArrow", CE_Failure);
    VALIDATE_POINTER1(out_schema, "GDALPolygonizeToArrow", CE_Failure);
    VALIDATE_POINTER1(out_array, "GDALPolygonizeToArrow", CE_Failure);
    const char *const pszNotSupported =
        "GDALPolygonizeToArrow(): not a request that is answered without a "
        "layer";
    if (!CPL_IS_LSB ||
        !CPLTestBool(CPLGetConfigOption("GDAL_POLYGONIZE_ARROW", "YES")) ||
        !TakesBand(hSrcBand, hMaskBand, papszOptions))
    {
        CPLError(CE_Failure, CPLE_NotSupported, "%s", pszNotSupported);
        return CE_Failure;
    }
    if (pfnProgress == nullptr)
        pfnProgress = GDALDummyProgress;

    ArrowJob oJob;
    oJob.bBandsAllowed = true;
    CPLErr eErr = CE_None;
    const bool bTaken =
        bFloat ? RunFloat(hSrcBand, nullptr, -1, papszOptions, pfnProgress,
                          pProgressArg, &eErr, &oJob)
               : RunInteger(hSrcBand, nullptr, -1, papszOptions, pfnProgress,
                            pProgressArg, &eErr, &oJob);
    if (!bTaken)
    {
        CPLError(CE_Failure, CPLE_NotSupported, "%s", pszNotSupported);
        return CE_Failure;
    }
    if (eErr != CE_None)
        return eErr;

    const auto oStart = std::chrono::steady_clock::now();
    struct ArrowArray sArray;
    bool bLarge = false;
    const int nRet = oJob.poBands ? WriteArrowArrayOfBands(oJob, &sArray, &bLarge)
                                  : WriteArrowArray(oJob, &sArray);
    if (nRet == EOVERFLOW)
    {
        CPLError(CE_Failure, CPLE_NotSupported,
                 "GDALPolygonizeToArrow(): more than 2 GB of WKB");
        return CE_Failure;
    }
    if (nRet != 0)
    {
        CPLError(CE_Failure, CPLE_OutOfMemory,
                 "GDALPolygonizeToArrow(): out of memory");
        return CE_Failure;
    }
    {
        const void *pOffsets = sArray.children[0]->buffers[1];
        const int64_t nWkbBytes =
            sArray.length == 0
                ? 0
                : bLarge ? static_cast<const int64_t *>(pOffsets)[sArray.length]
                         : static_cast<const int32_t *>(pOffsets)[sArray.length];
        CPLDebug("GDAL",
                 "X-Rasp: seconds: Arrow array written %.2f (" CPL_FRMT_GIB
                 " polygons, %.2f GB of WKB%s)",
                 Seconds(oStart), static_cast<GIntBig>(sArray.length),
                 static_cast<double>(nWkbBytes) / 1e9,
                 bLarge ? ", offsets of 64 bits" : "");
    }
    pfnProgress(1.0, "", pProgressArg);
    *out_array = sArray;
    InitArrowSchema(out_schema, bLarge);
    return CE_None;
}
