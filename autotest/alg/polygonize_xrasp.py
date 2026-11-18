#!/usr/bin/env pytest
###############################################################################
#
# Project:  GDAL/OGR Test Suite
# Purpose:  Polygonize() with X-Rasp, the parallel polygonizer (alg/xrasp):
#           the layer it writes is the layer the default polygonizer writes,
#           the same features in the same order, point for point.
#
###############################################################################
# SPDX-License-Identifier: MIT
###############################################################################

import random

import pytest

from osgeo import gdal, ogr

np = pytest.importorskip("numpy")

GDT = {
    "u1": gdal.GDT_Byte,
    "i1": gdal.GDT_Int8,
    "u2": gdal.GDT_UInt16,
    "i2": gdal.GDT_Int16,
    "u4": gdal.GDT_UInt32,
    "i4": gdal.GDT_Int32,
    "i8": gdal.GDT_Int64,
    "u8": gdal.GDT_UInt64,
    "f4": gdal.GDT_Float32,
    "f8": gdal.GDT_Float64,
}

GP_NODATA_MARKER = -51502112


def raster_of(cells, geotransform=None):
    """A MEM dataset with one band holding the 2-D numpy array."""
    h, w = cells.shape
    ds = gdal.GetDriverByName("MEM").Create("", w, h, 1, GDT[cells.dtype.str[1:]])
    if geotransform is not None:
        ds.SetGeoTransform(geotransform)
    ds.GetRasterBand(1).WriteRaster(0, 0, w, h, cells.tobytes())
    return ds


def polygonize(
    band,
    xrasp,
    mask=None,
    options=(),
    threads=None,
    field_type=ogr.OFTInteger64,
    field=0,
    layer=None,
    callback=None,
    expect_failure=False,
):
    """Run Polygonize() with X-Rasp allowed or not.

    Returns (features, taken): the features as (value, WKB) in the order of
    the layer, and whether X-Rasp took the request.
    """
    out_ds = None
    if layer is None:
        out_ds = gdal.GetDriverByName("MEM").Create("", 0, 0, 0, gdal.GDT_Unknown)
        layer = out_ds.CreateLayer("poly", None, ogr.wkbPolygon)
        layer.CreateField(ogr.FieldDefn("DN", field_type))
    messages = []

    def handler(err_class, err_no, message):
        messages.append(message)

    config = {"GDAL_POLYGONIZE_XRASP": "YES" if xrasp else "NO", "CPL_DEBUG": "GDAL"}
    if threads is not None:
        config["GDAL_NUM_THREADS"] = str(threads)
    gdal.PushErrorHandler(handler)
    try:
        with gdal.config_options(config):
            result = gdal.Polygonize(
                band, mask, layer, field, list(options), callback=callback
            )
        failed = result != 0
    except RuntimeError as e:
        messages.append(str(e))
        failed = True
    finally:
        gdal.PopErrorHandler()
    assert failed == expect_failure, messages
    if failed:
        return None, False
    taken = any("X-Rasp: " in m and " polygons with " in m for m in messages)
    layer.ResetReading()
    features = [
        (f.GetField(0) if field >= 0 else None, bytes(f.GetGeometryRef().ExportToWkb()))
        for f in layer
    ]
    return features, taken


def built_with_xrasp():
    cells = np.zeros((2, 2), dtype=np.uint8)
    ds = raster_of(cells)
    _, taken = polygonize(ds.GetRasterBand(1), True)
    return taken


pytestmark = pytest.mark.skipif(
    not built_with_xrasp(), reason="GDAL built without X-Rasp"
)


def describe(features, i):
    value, wkb = features[i]
    return "%s %s" % (value, ogr.CreateGeometryFromWkb(wkb).ExportToWkt()[:300])


def check_same(cells, geotransform=None, expect_taken=True, **kwargs):
    """X-Rasp and the default polygonizer write the same layer."""
    ds = raster_of(cells, geotransform)
    band = ds.GetRasterBand(1)
    expected, taken_off = polygonize(band, False, **kwargs)
    got, taken = polygonize(band, True, **kwargs)
    assert not taken_off
    assert taken == expect_taken
    assert len(got) == len(expected)
    for i in range(len(got)):
        assert got[i] == expected[i], "feature %d of %d: %s, expected %s" % (
            i,
            len(got),
            describe(got, i),
            describe(expected, i),
        )
    return got


def random_cells(rng, h, w, dtype, values, p_first):
    """Cells drawn from `values`, the first one with probability p_first."""
    pick = rng.random((h, w)) < p_first
    other = rng.integers(1, len(values), size=(h, w)) if len(values) > 1 else 0
    index = np.where(pick, 0, other)
    return np.asarray(values, dtype=dtype)[index]


def blobs(rng, h, w, levels, smooth):
    """Smooth random regions: noise averaged over a window, cut into levels."""
    noise = rng.random((h + 2 * smooth, w + 2 * smooth))
    total = noise.cumsum(axis=0).cumsum(axis=1)
    k = 2 * smooth
    box = total[k:, k:] - total[:-k, k:] - total[k:, :-k] + total[:-k, :-k]
    box = box[:h, :w]
    cuts = np.quantile(box, np.linspace(0, 1, levels + 1)[1:-1])
    return np.searchsorted(cuts, box).astype(np.uint8)


###############################################################################
# Small rasters of every cell type X-Rasp takes


@pytest.mark.parametrize("dtype", ["u1", "i1", "u2", "i2", "u4", "i4"])
def test_polygonize_xrasp_small_random(dtype):
    rng = np.random.default_rng(1)
    info = np.iinfo(np.dtype(dtype))
    pool = [0, 1, info.max, info.min, info.max - 1, 7, 100]
    count = 0
    for values in (2, 3, 5):
        for p_first in (0.3, 0.5, 0.7):
            for _ in range(40):
                h, w = (int(v) for v in rng.integers(1, 21, size=2))
                check_same(random_cells(rng, h, w, dtype, pool[:values], p_first))
                count += 1
    assert count == 360


###############################################################################
# Shapes that are easy to get wrong


def test_polygonize_xrasp_shapes():
    one = np.ones((1, 1), dtype=np.uint8)
    check_same(one)
    check_same(np.zeros((1, 37), dtype=np.uint8))
    check_same(np.zeros((37, 1), dtype=np.uint8))
    check_same(np.zeros((40, 50), dtype=np.uint8))
    check_same(np.array([[1, 1, 2, 2, 1, 3, 3, 3, 1]], dtype=np.uint8))
    check_same(np.array([[1, 1, 2, 2, 1, 3, 3, 3, 1]], dtype=np.uint8).T)
    # every cell its own polygon
    check_same(np.arange(30 * 41, dtype=np.uint16).reshape(30, 41))
    # checkerboards: cells of one value touch only at corners
    y, x = np.mgrid[0:33, 0:47]
    check_same(((x + y) % 2).astype(np.uint8))
    check_same(((x // 2 + y // 3) % 2).astype(np.uint8))
    # stripes, and a comb whose teeth are joined by its last row only
    check_same((x % 2).astype(np.uint8))
    check_same((y % 2).astype(np.uint8))
    comb = (x % 2).astype(np.uint8)
    comb[-1, :] = 1
    check_same(comb)
    comb = (x % 2).astype(np.uint8)
    comb[0, :] = 1
    check_same(comb)
    # squares inside squares
    rings = np.minimum(np.minimum(x, 46 - x), np.minimum(y, 32 - y))
    check_same((rings % 2).astype(np.uint8))
    check_same((rings % 3).astype(np.uint8))
    # diagonal bands: staircases, and holes that touch at corners
    check_same((((x + y) // 3) % 2).astype(np.uint8))
    check_same((((x - y) // 3) % 2).astype(np.uint8))
    check_same(((x * y) % 3 == 0).astype(np.uint8))
    # a spiral: one long polygon that turns around itself
    spiral = np.zeros((41, 41), dtype=np.uint8)
    cy, cx, step, d = 20, 20, 2, 0
    for leg in range(1, 40):
        for _ in range(step * ((leg + 1) // 2)):
            if 0 <= cy < 41 and 0 <= cx < 41:
                spiral[cy, cx] = 1
            cy += (0, 1, 0, -1)[d]
            cx += (1, 0, -1, 0)[d]
        d = (d + 1) % 4
    check_same(spiral)


###############################################################################
# GDAL's own example rasters


@pytest.mark.require_driver("AAIGRID")
@pytest.mark.parametrize(
    "name",
    [
        "polygonize_in.grd",
        "polygonize_in_2.grd",
        "polygonize_in_3.grd",
        "polygonize_in_4.grd",
        "polygonize_in_5.grd",
    ],
)
def test_polygonize_xrasp_examples(name):
    src_ds = gdal.Open("data/" + name)
    band = src_ds.GetRasterBand(1)
    expected, _ = polygonize(band, False, field_type=ogr.OFTInteger)
    got, taken = polygonize(band, True, field_type=ogr.OFTInteger)
    assert taken
    assert got == expected


###############################################################################
# Rasters large enough for every step to run on several threads, with
# different numbers of threads


@pytest.mark.parametrize("threads", [1, 2, 3, 4, 7, None])
def test_polygonize_xrasp_large_random(threads):
    rng = np.random.default_rng(2)
    for values, p_first in ((2, 0.4), (2, 0.5), (2, 0.6), (3, 0.45), (5, 0.3)):
        cells = random_cells(rng, 331, 517, "u1", list(range(values)), p_first)
        check_same(cells, threads=threads)


@pytest.mark.parametrize("threads", [1, 4, None])
def test_polygonize_xrasp_large_regions(threads):
    rng = np.random.default_rng(3)
    for levels, smooth, shape in ((2, 3, (900, 1100)), (4, 6, (1200, 1000)), (7, 2, (700, 1300))):
        check_same(blobs(rng, shape[0], shape[1], levels, smooth), threads=threads)


###############################################################################
# Georeferencing: north up, south up, rotated


def test_polygonize_xrasp_geotransform():
    rng = np.random.default_rng(4)
    cells = random_cells(rng, 23, 31, "u1", [0, 1, 2], 0.5)
    for gt in (
        None,
        (440720.0, 60.0, 0.0, 3751320.0, 0.0, -60.0),
        (-180.0, 0.25, 0.0, -90.0, 0.0, 0.25),
        (100.0, 0.8, 0.6, 200.0, -0.6, 0.8),
        (1e-7, 1e-9, 0.0, 5e6, 0.0, -1.0 / 3.0),
    ):
        check_same(cells, geotransform=gt)


###############################################################################
# Values: the largest and smallest of a type, the attribute as an integer, a
# real number, or not written; the value the default polygonizer keeps for
# masked pixels


def test_polygonize_xrasp_values():
    rng = np.random.default_rng(5)
    cells = random_cells(rng, 19, 27, "u4", [0, 4294967295, 2147483648, 5], 0.4)
    got = check_same(cells)
    assert max(v for v, _ in got) == 4294967295
    check_same(cells, field_type=ogr.OFTReal)
    check_same(cells, field=-1)
    cells = random_cells(rng, 19, 27, "i4", [-2147483648, 2147483647, -1, 0], 0.4)
    got = check_same(cells)
    assert min(v for v, _ in got) == -2147483648
    cells = random_cells(rng, 19, 27, "i1", [-128, 127, -1, 0], 0.4)
    got = check_same(cells)
    assert min(v for v, _ in got) == -128


def test_polygonize_xrasp_marker_value():
    rng = np.random.default_rng(6)
    for _ in range(20):
        cells = random_cells(rng, 17, 21, "i4", [3, GP_NODATA_MARKER, 5], 0.5)
        got = check_same(cells)
        assert all(v != GP_NODATA_MARKER for v, _ in got)
    check_same(np.full((5, 6), GP_NODATA_MARKER, dtype=np.int32))


###############################################################################
# Requests X-Rasp leaves to the default polygonizer


def test_polygonize_xrasp_not_taken():
    rng = np.random.default_rng(7)
    cells = random_cells(rng, 25, 33, "u1", [0, 1, 2], 0.5)
    ds = raster_of(cells)
    band = ds.GetRasterBand(1)
    expected, _ = polygonize(band, False)

    got, taken = polygonize(band, True, options=["8CONNECTED=8"])
    assert not taken
    assert got == polygonize(band, False, options=["8CONNECTED=8"])[0]

    mask_ds = raster_of(random_cells(rng, 25, 33, "u1", [1, 0], 0.7))
    mask = mask_ds.GetRasterBand(1)
    got, taken = polygonize(band, True, mask=mask)
    assert not taken
    assert got == polygonize(band, False, mask=mask)[0]

    for dtype in ("i8", "u8", "f4", "f8"):
        other = raster_of(cells.astype(dtype))
        got, taken = polygonize(other.GetRasterBand(1), True)
        assert not taken
        assert got == expected


###############################################################################
# A layer that already has features; the commit interval


def test_polygonize_xrasp_layer():
    rng = np.random.default_rng(8)
    cells = random_cells(rng, 25, 33, "u1", [0, 1, 2], 0.5)
    ds = raster_of(cells)
    band = ds.GetRasterBand(1)
    expected, _ = polygonize(band, False)

    out_ds = gdal.GetDriverByName("MEM").Create("", 0, 0, 0, gdal.GDT_Unknown)
    layer = out_ds.CreateLayer("poly", None, ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("DN", ogr.OFTInteger64))
    twice, taken = polygonize(band, True, layer=layer)
    assert taken
    twice, taken = polygonize(band, True, layer=layer)
    assert taken
    assert twice == expected + expected

    for interval in ("0", "1", "7", "-1"):
        check_same(cells, options=["COMMIT_INTERVAL=" + interval])


@pytest.mark.require_driver("GPKG")
def test_polygonize_xrasp_gpkg(tmp_vsimem):
    rng = np.random.default_rng(9)
    cells = random_cells(rng, 40, 50, "u1", [0, 1, 2], 0.5)
    ds = raster_of(cells, (440720.0, 60.0, 0.0, 3751320.0, 0.0, -60.0))
    band = ds.GetRasterBand(1)
    layers = []
    for xrasp in (False, True):
        for interval in ("100000", "5"):
            name = str(tmp_vsimem / ("out_%d_%s.gpkg" % (xrasp, interval)))
            out_ds = gdal.GetDriverByName("GPKG").Create(name, 0, 0, 0, gdal.GDT_Unknown)
            layer = out_ds.CreateLayer("poly", None, ogr.wkbPolygon)
            layer.CreateField(ogr.FieldDefn("DN", ogr.OFTInteger64))
            features, taken = polygonize(
                band, xrasp, layer=layer, options=["COMMIT_INTERVAL=" + interval]
            )
            assert taken == xrasp
            out_ds = None
            out_ds = ogr.Open(name)
            layer = out_ds.GetLayer(0)
            layers.append(
                [(f.GetFID(), f.GetField(0), bytes(f.GetGeometryRef().ExportToWkb())) for f in layer]
            )
    assert len(layers[0]) > 100
    for other in layers[1:]:
        assert other == layers[0]


###############################################################################
# A band that is read from a file in blocks


def test_polygonize_xrasp_gtiff(tmp_vsimem):
    rng = np.random.default_rng(10)
    cells = blobs(rng, 300, 420, 4, 4)
    name = str(tmp_vsimem / "in.tif")
    src = raster_of(cells, (440720.0, 60.0, 0.0, 3751320.0, 0.0, -60.0))
    src.GetRasterBand(1).SetNoDataValue(2)
    gdal.GetDriverByName("GTiff").CreateCopy(
        name, src, options=["TILED=YES", "BLOCKXSIZE=64", "BLOCKYSIZE=64", "COMPRESS=DEFLATE"]
    )
    ds = gdal.Open(name)
    band = ds.GetRasterBand(1)
    expected, _ = polygonize(band, False)
    got, taken = polygonize(band, True)
    assert taken
    assert got == expected
    # the nodata value of the band is not a mask unless it is passed as one
    assert any(v == 2 for v, _ in got)
    # the geotransform of another dataset
    plain = raster_of(cells)
    option = ["DATASET_FOR_GEOREF=" + name]
    got, taken = polygonize(plain.GetRasterBand(1), True, options=option)
    assert taken
    assert got == expected


###############################################################################
# Progress and interruption


def test_polygonize_xrasp_progress():
    rng = np.random.default_rng(11)
    cells = random_cells(rng, 60, 80, "u1", [0, 1, 2], 0.5)
    ds = raster_of(cells)
    band = ds.GetRasterBand(1)
    seen = []

    def progress(fraction, message, data):
        seen.append(fraction)
        return 1

    _, taken = polygonize(band, True, callback=progress)
    assert taken
    assert seen == sorted(seen)
    assert seen[-1] == 1.0

    def stop(fraction, message, data):
        return 0

    with gdal.quiet_errors():
        polygonize(band, True, callback=stop, expect_failure=True)
        polygonize(band, False, callback=stop, expect_failure=True)


###############################################################################
# Many random rasters of mixed sizes, each with a number of threads of its own


def test_polygonize_xrasp_many():
    rng = np.random.default_rng(12)
    py_rng = random.Random(12)
    for i in range(150):
        h, w = (int(v) for v in rng.integers(1, 160, size=2))
        dtype = py_rng.choice(["u1", "i1", "u2", "i2", "u4", "i4"])
        values = py_rng.choice([2, 2, 3, 4, 9])
        threads = py_rng.choice([1, 2, 3, 4, 5, 8])
        if py_rng.random() < 0.3:
            cells = blobs(rng, h, w, values, py_rng.choice([1, 2, 4])).astype(dtype)
        else:
            cells = random_cells(rng, h, w, dtype, list(range(values)), py_rng.choice([0.2, 0.4, 0.5, 0.6, 0.8]))
        check_same(cells, threads=threads)
