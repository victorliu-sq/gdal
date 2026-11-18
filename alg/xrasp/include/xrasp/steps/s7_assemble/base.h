#ifndef XRASP_STEPS_S7_ASSEMBLE_BASE_H
#define XRASP_STEPS_S7_ASSEMBLE_BASE_H
// Step 7: rings -> polygons, and the vertices into their final places.
//
// The rings are ordered by their outer ring, the outer ring first (holes keep the order of their ring
// numbers).  A polygon is one outer ring with its holes.  Every vertex then goes to
// (start of its ring in that order) + (its position in the ring).
//
// With a ScanOrder (opt_scan_order.h) the polygons, the holes of a polygon and the vertices of a ring come
// in the order of GDAL's polygonizer instead.
#include <cstdint>

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "xrasp/steps/s7_assemble/opt_scan_order.h"
#include "xrasp/structs/polygons.h"
#include "xrasp/structs/rings.h"
#include "xrasp/util/dev_array.h"
#include "xrasp/util/launcher.h"

namespace xrasp {

struct IsPolygonStart {
  const int64_t* key;      // outer ring * 2 + is_hole, the rings of a polygon together
  DEV_HOST_INLINE bool operator()(int64_t i) const { return i == 0 || (key[i] >> 1) != (key[i - 1] >> 1); }
};

// ring_value[r]: the raster value of ring r.  vx, vy: the corner of every vertex.  scan: null, or the order
// of GDAL's polygonizer; without the numbers of the polygons in it (scan->seed empty) the polygons that end in
// one row come in the order of their outer rings.  outer: null, or where to put the outer ring of every
// polygon.
inline void AssemblePolygons(const Rings& rings, const DArray<int64_t>& ring_value, const DArray<int32_t>& vx,
                             const DArray<int32_t>& vy, Polygons* out, const ScanOrder* scan = nullptr,
                             DArray<int32_t>* outer = nullptr) {
  const int64_t nr = rings.nr, nv = rings.nv;
  DArray<int64_t> key(nr);
  DArray<int32_t> order(nr);
  thrust::sequence(Exec(), order.begin(), order.end(), 0);
  if (scan != nullptr) {
    // three stable sorts, the least significant key first: the hole's top-left vertex (0 for the outer
    // ring), the polygon's number (or its outer ring), the polygon's last row
    auto k = key.DeviceView();
    auto ord = order.DeviceView();
    auto parent = rings.parent.DeviceView();
    auto hole = rings.is_hole.DeviceView();
    auto first = scan->first.DeviceView(), seed = scan->seed.DeviceView(), last_row = scan->last_row.DeviceView();
    ForEach(nr, [=] __device__(size_t r) { k[r] = hole[r] ? first[r] + 1 : 0; });
    thrust::stable_sort_by_key(Exec(), key.begin(), key.end(), order.begin());
    const bool numbered = !scan->seed.empty();
    ForEach(nr, [=] __device__(size_t i) { k[i] = numbered ? seed[parent[ord[i]]] : parent[ord[i]]; });
    thrust::stable_sort_by_key(Exec(), key.begin(), key.end(), order.begin());
    ForEach(nr, [=] __device__(size_t i) { k[i] = last_row[parent[ord[i]]]; });
    thrust::stable_sort_by_key(Exec(), key.begin(), key.end(), order.begin());
    // the key the rest reads: outer ring * 2 + is_hole, equal outer rings being neighbours
    ForEach(nr, [=] __device__(size_t i) { k[i] = static_cast<int64_t>(parent[ord[i]]) * 2 + hole[ord[i]]; });
  } else {
    {
      auto k = key.DeviceView();
      auto parent = rings.parent.DeviceView();
      auto hole = rings.is_hole.DeviceView();
      ForEach(nr, [=] __device__(size_t r) { k[r] = static_cast<int64_t>(parent[r]) * 2 + hole[r]; });
    }
    thrust::stable_sort_by_key(Exec(), key.begin(), key.end(), order.begin());
  }

  DArray<int64_t> start_all(nr);
  int64_t* start_end = thrust::copy_if(Exec(), thrust::make_counting_iterator<int64_t>(0),
                                       thrust::make_counting_iterator<int64_t>(nr), start_all.begin(),
                                       IsPolygonStart{key.data()});
  const int64_t npoly = start_end - start_all.begin();
  key.Free();
  out->poly_offsets.Reset(npoly + 1);
  out->values.Reset(npoly);
  if (outer != nullptr) {
    outer->Reset(npoly);
    auto s = start_all.DeviceView();
    auto ord = order.DeviceView();
    auto o = outer->DeviceView();
    ForEach(npoly, [=] __device__(size_t p) { o[p] = ord[s[p]]; });
  }
  {
    auto s = start_all.DeviceView();
    auto po = out->poly_offsets.DeviceView();
    auto val = out->values.DeviceView();
    auto ord = order.DeviceView();
    auto rv = ring_value.DeviceView();
    ForEach(npoly + 1, [=] __device__(size_t p) {
      if (static_cast<int64_t>(p) == npoly) {
        po[p] = nr;
      } else {
        po[p] = s[p];
        val[p] = rv[ord[s[p]]];
      }
    });
  }
  start_all.Free();

  // ring_offsets[i]: where the i-th ring of the final order starts
  out->ring_offsets.Reset(nr + 1);
  DArray<int32_t> slot(nr);           // place of ring r in the final order
  {
    auto ord = order.DeviceView();
    auto len = rings.length.DeviceView();
    auto ro = out->ring_offsets.DeviceView();
    auto sl = slot.DeviceView();
    ForEach(nr + 1, [=] __device__(size_t i) {
      if (i == 0) {
        ro[0] = 0;
      }
      if (static_cast<int64_t>(i) < nr) {
        ro[i + 1] = len[ord[i]];
        sl[ord[i]] = static_cast<int32_t>(i);
      }
    });
    thrust::inclusive_scan(Exec(), out->ring_offsets.begin(), out->ring_offsets.end(), out->ring_offsets.begin());
  }
  order.Free();

  out->x.Reset(nv);
  out->y.Reset(nv);
  {
    auto ro = out->ring_offsets.DeviceView();
    auto sl = slot.DeviceView();
    auto ring = rings.ring.DeviceView();
    auto rank = rings.rank.DeviceView();
    auto x = vx.DeviceView(), y = vy.DeviceView();
    auto ox = out->x.DeviceView(), oy = out->y.DeviceView();
    if (scan != nullptr) {
      // the ring starts at its top-left vertex
      auto len = rings.length.DeviceView();
      auto fr = scan->first_rank.DeviceView();
      ForEach(nv, [=] __device__(size_t v) {
        int32_t r = ring[v];
        int64_t at = rank[v] - fr[r];
        if (at < 0) at += len[r];
        int64_t dest = ro[sl[r]] + at;
        ox[dest] = x[v];
        oy[dest] = y[v];
      });
    } else {
      ForEach(nv, [=] __device__(size_t v) {
        int64_t dest = ro[sl[ring[v]]] + rank[v];
        ox[dest] = x[v];
        oy[dest] = y[v];
      });
    }
  }
}

}  // namespace xrasp
#endif  // XRASP_STEPS_S7_ASSEMBLE_BASE_H
