#ifndef XRASP_STEPS_S6_HOLES_BY_WEST_SEARCH_H
#define XRASP_STEPS_S6_HOLES_BY_WEST_SEARCH_H
// Step 6 (no block labels): which outer ring does a hole belong to?
//
// From the cell just west of the hole's leftmost edge -- a cell of the block the hole is in -- go west along
// the row until the value changes.  The unit edge found there is on another ring of the same block, walked
// downwards with the block on its east; its straight run ends at the first vertex entered moving down in
// that column at or below the next row.  Both lookups are one binary search: in the sorted list of vertical
// unit edges, then in the sorted list of down-entered vertices.  That gives every hole one ring of its
// block; following these links (pointer jumping over rings) ends at the block's outer ring.
#include <cstdint>

#include <thrust/sequence.h>

#include "xrasp/structs/rings.h"
#include "xrasp/structs/vertex_lists.h"
#include "xrasp/util/dev_array.h"
#include "xrasp/util/dev_value.h"
#include "xrasp/backend/launcher.h"

namespace xrasp {

// edges: the vertical unit edges of the raster, sorted positions cy * wc + x.  Fills rings->parent and
// returns the number of jumping rounds.
inline int AttachHolesByWestSearch(const VertexLists& lists, const DArray<int64_t>& edges, Rings* rings) {
  const int64_t nr = rings->nr;
  rings->parent.Reset(nr);
  thrust::sequence(Exec(), rings->parent.begin(), rings->parent.end(), 0);
  if (rings->nholes == 0) return 0;
  const int64_t wc = lists.wc, hc = lists.hc;
  const int64_t ne = static_cast<int64_t>(edges.size());
  {
    auto e = edges.DeviceView();
    auto L = lists.DeviceView();
    auto hs = rings->holes.DeviceView();
    auto cx = rings->hole_cx.DeviceView(), cy = rings->hole_cy.DeviceView();
    auto ring = rings->ring.DeviceView();
    auto parent = rings->parent.DeviceView();
    ForEach(rings->nholes, [=] __device__(size_t i) {
      int64_t at = UpperBound(e.data(), ne, cy[i] * wc + cx[i]) - 1;      // the last edge at or west of the cell
      int64_t xh = e[at] % wc;
      int64_t key = xh * hc + (cy[i] + 1);
      int64_t u = LowerBound(L.pos[0], L.count[0], key);                  // base[0] is 0
      parent[hs[i]] = ring[u];
    });
  }
  // hole -> ring -> ... -> outer ring
  DArray<int32_t> next(nr);
  DValue<int> changed;
  int hops = 0;
  while (true) {
    changed.SetH2D(0);
    auto p = rings->parent.DeviceView();
    auto q = next.DeviceView();
    auto flag = changed.DeviceView();
    ForEach(nr, [=] __device__(size_t r) {
      int32_t pp = p[p[r]];
      q[r] = pp;
      if (pp != p[r]) flag.Set(1);
    });
    if (changed.GetD2H() == 0) break;
    rings->parent.Swap(next);
    hops++;
  }
  return hops;
}

}  // namespace xrasp
#endif  // XRASP_STEPS_S6_HOLES_BY_WEST_SEARCH_H
