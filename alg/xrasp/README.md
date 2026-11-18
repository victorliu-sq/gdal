# X-Rasp inside GDAL

`GDALPolygonize()` runs X-Rasp, a parallel polygonizer, when it can
(`alg/polygonize_xrasp.cpp`), and the polygonizer GDAL always had otherwise.

X-Rasp writes the same features in the same order -- the same polygons, their
holes in the same order, every ring from the same vertex -- found with all the
cores of the machine: every vertex of every outline is found on its own, finds
its next vertex with a search, and the outlines are ordered by list ranking.

The order is the default polygonizer's by construction, not by a sort of its
output (`include/xrasp/steps/s7_assemble/opt_scan_order.h`): a polygon is
written after the row of its last cell; polygons that end in the same row
follow the numbers the default polygonizer gives them in its first scan, which
X-Rasp finds from the vertices; a ring starts at its top-left vertex.

It takes a request when

* there is no mask band and 4 connectedness is asked for (no `8CONNECTED`),
* the band is of an integer type of at most 32 bits (`GDALPolygonize()`), or
  not of type Float64 and without neighbouring cells that are almost equal
  but not the same float (`GDALFPolygonize()`, see below),
* the whole band fits in memory (the default polygonizer reads line by line).

X-Rasp is standard C++, part of every build of GDAL, and needs nothing GDAL
does not have. Its threads are OpenMP's when GDAL itself uses OpenMP
(`GDAL_USE_OPENMP`, on wherever OpenMP is found), and GDAL's own worker
threads (`GDALGetGlobalThreadPool()`) otherwise. The features are the same with either.

Otherwise, and when X-Rasp gives up (more than 2^31 - 1 vertices), the default
polygonizer runs; nothing has been written by then.

`GDALFPolygonize()` takes two neighbouring cells for equal when at most 10
floats lie between them. That is not transitive, and the default polygonizer
draws no edge between two cells of one polygon whatever their values, while
X-Rasp draws an edge wherever two neighbours differ. The two agree when every
pair of neighbours is either the same float or further apart than that, which
X-Rasp checks first, with all the threads; it then runs on the bits of the
floats. Rasters of classes kept as floats pass; a raster where two
neighbours are almost equal and not the same is left to the default
polygonizer.

Configuration options:

* `GDAL_POLYGONIZE_XRASP=NO` turns it off.
* `GDAL_NUM_THREADS=n` or `ALL_CPUS` sets the number of threads. The default
  is the number of physical cores: the work is bound by memory, and a second
  thread on a core does not help.

## How it is checked

`autotest/alg/polygonize_xrasp.py` calls `Polygonize()` twice, with X-Rasp and
with the default polygonizer, and compares the two layers feature by feature
in order, value and WKB: random rasters of every cell type X-Rasp takes,
shapes that are easy to get wrong (checkerboards, combs, spirals, holes that
touch at corners), GDAL's example rasters, large rasters with 1 to 7 threads,
geotransforms, the extreme values of the types and `GP_NODATA_MARKER`, a layer
that already has features, GeoPackage with transactions, a tiled GeoTIFF,
progress and interruption; and it checks that the requests X-Rasp does not
take reach the default polygonizer. GDAL's own tests that polygonize
(`autotest/alg/polygonize.py`, `autotest/pyscripts/test_gdal_polygonize.py`,
`autotest/utilities/test_gdalalg_raster_polygonize.py`,
`autotest/utilities/test_gdal_footprint_lib.py`) pass unchanged with
`GDAL_POLYGONIZE_XRASP` at `YES` and at `NO`.

On whole rasters the layers were compared with a hash of the layer as it is
read (every value, ring and point in order; the driver of the X-Rasp
repository, `Result.Sequence`), 2026-11-17, 32 cores:

| raster | polygons | points | threads | layer |
|---|---|---|---|---|
| Zones (30 arc-second) | 100 955 | 7 900 028 | 1, 7, 32, 64 | the same |
| Global Surface Water | 673 197 | 18 473 788 | 1, 7, 32, 64 | the same |
| WorldCover tile | 3 854 665 | 116 521 568 | 32 | the same |
| CORINE | 4 412 020 | 191 517 282 | 32 | the same |

## Where the code comes from

`include/`, `shim/` and `device_memory.cpp` are copies of the CPU version of
the X-Rasp library and are not edited here: they are written by
`src/targets/scripts/sync_gdal` of the X-Rasp repository, and `SYNCED_FROM`
holds the commit they were taken at. `shim/thrust` stands in for Thrust, whose
names the steps use; the work behind them is in
`include/xrasp/util/cpu_primitives.h`, on the threads of
`include/xrasp/util/cpu_threads.h`, so nothing else is needed.
