#ifndef XRASP_STEPS_BANDS_B2_BAND_HOLES_H
#define XRASP_STEPS_BANDS_B2_BAND_HOLES_H
// Band step 2 (mode=bands): attach the holes of the rings finished inside a band.
//
// The search is the one of s6_holes/by_west_search.h: west along the row to the first vertical unit edge, then
// down that column to the first vertex entered moving down.  That vertex is looked for among ALL down-entered
// vertices of the band.  If it was kept (it is on a ring finished in this band), the hole is attached to that
// ring.  If it was set aside, or the run leaves the band, the hole WAITS: its search key is remembered and
// answered at the end from the set-aside vertices.
//
// Ring numbers are global here: ring_base + (number inside the band).
#include <cstdint>

#include "xrasp/structs/rings.h"
#include "xrasp/util/algo.h"
#include "xrasp/util/dev_array.h"
#include "xrasp/backend/launcher.h"

namespace xrasp {

// pos0_all: the down-entered vertices of the whole band (sorted); owner[i]: index of vertex i among the kept
// ones, or -1.  The kept down-entered vertices are vertices 0 .. of the band's own lists, so their rings are
// rings.ring[owner].  Fills parent (global ring numbers) and pending (the search key, or -1).
inline void AttachBandHoles(const Rings& rings, const DArray<int64_t>& edges, const DArray<int64_t>& pos0_all,
                            const DArray<int32_t>& owner, int64_t wc, int64_t hc, int64_t ring_base,
                            DArray<int64_t>* parent, DArray<int64_t>* pending) {
  const int64_t nr = rings.nr;
  parent->Reset(nr);
  pending->Reset(nr);
  {
    auto p = parent->DeviceView(), w = pending->DeviceView();
    ForEach(nr, [=] __device__(size_t r) {
      p[r] = ring_base + static_cast<int64_t>(r);
      w[r] = -1;
    });
  }
  if (rings.nholes == 0) return;
  const int64_t ne = static_cast<int64_t>(edges.size());
  const int64_t n0 = static_cast<int64_t>(pos0_all.size());
  auto e = edges.DeviceView();
  auto p0 = pos0_all.DeviceView();
  auto own = owner.DeviceView();
  auto hs = rings.holes.DeviceView();
  auto cx = rings.hole_cx.DeviceView(), cy = rings.hole_cy.DeviceView();
  auto ring = rings.ring.DeviceView();
  auto p = parent->DeviceView(), w = pending->DeviceView();
  ForEach(rings.nholes, [=] __device__(size_t i) {
    int64_t at = UpperBound(e.data(), ne, cy[i] * wc + cx[i]) - 1;
    int64_t key = (e[at] % wc) * hc + (cy[i] + 1);
    int64_t k = LowerBound(p0.data(), n0, key);
    bool found = k < n0 && p0[k] / hc == key / hc && own[k] >= 0;      // the run ends at a kept vertex
    if (found) {
      p[hs[i]] = ring_base + ring[own[k]];
    } else {
      w[hs[i]] = key;                                // it ends at a set-aside vertex, or below the band
    }
  });
}

// The vertices of the finished rings in ring order: ring 0's vertices, then ring 1's, ...
inline void VerticesInRingOrder(const Rings& rings, const DArray<int32_t>& vx, const DArray<int32_t>& vy,
                                DArray<int32_t>* xr, DArray<int32_t>* yr) {
  DArray<int64_t> start(rings.nr);
  {
    auto len = rings.length.DeviceView();
    auto s = start.DeviceView();
    ForEach(rings.nr, [=] __device__(size_t r) { s[r] = len[r]; });
  }
  thrust::exclusive_scan(Exec(), start.begin(), start.end(), start.begin());
  xr->Reset(rings.nv);
  yr->Reset(rings.nv);
  auto s = start.DeviceView();
  auto ring = rings.ring.DeviceView(), rank = rings.rank.DeviceView();
  auto x = vx.DeviceView(), y = vy.DeviceView();
  auto ox = xr->DeviceView(), oy = yr->DeviceView();
  ForEach(rings.nv, [=] __device__(size_t v) {
    int64_t dest = s[ring[v]] + rank[v];
    ox[dest] = x[v];
    oy[dest] = y[v];
  });
}

}  // namespace xrasp
#endif  // XRASP_STEPS_BANDS_B2_BAND_HOLES_H
