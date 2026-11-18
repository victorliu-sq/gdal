/******************************************************************************
 * Project:  GDAL
 * Purpose:  GDALPolygonize() with X-Rasp, a parallel polygonizer
 *
 ******************************************************************************
 * SPDX-License-Identifier: MIT
 ****************************************************************************/

#ifndef POLYGONIZE_XRASP_H_INCLUDED
#define POLYGONIZE_XRASP_H_INCLUDED

/*! @cond Doxygen_Suppress */

#include "cpl_error.h"
#include "cpl_port.h"
#include "cpl_progress.h"
#include "gdal.h"
#include "ogr_api.h"

/** Polygonize hSrcBand into hOutLayer with X-Rasp (alg/xrasp/README.md).
 *
 * Returns false when X-Rasp does not take the request -- a mask band, 8
 * connectedness, a cell type it does not read, GDAL_POLYGONIZE_XRASP=NO, not
 * enough memory for the whole band -- and nothing has been written: the caller
 * then runs the default polygonizer.  Returns true when it took the request;
 * *peErr is then the result.
 */
bool GDALPolygonizeXRasp(GDALRasterBandH hSrcBand, GDALRasterBandH hMaskBand,
                         OGRLayerH hOutLayer, int iPixValField,
                         CSLConstList papszOptions,
                         GDALProgressFunc pfnProgress, void *pProgressArg,
                         CPLErr *peErr);

/*! @endcond */

#endif /* POLYGONIZE_XRASP_H_INCLUDED */
