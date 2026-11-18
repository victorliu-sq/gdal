#ifndef XRASP_STEPS_S2_LINKS_BASE_H
#define XRASP_STEPS_S2_LINKS_BASE_H
// Step 2: the next vertex of every vertex.
//
// After its turn the outline runs straight until the next vertex, which may be far away.  Following the run
// cell by cell is what makes tracing sequential; here every vertex finds its successor on its own, with one
// binary search in the sorted list of the direction it leaves in (VertexListsView::Next).
#include <cstdint>

#include "xrasp/structs/vertex_lists.h"
#include "xrasp/util/dev_array.h"
#include "xrasp/util/launcher.h"

namespace xrasp {

// nxt[v]: the vertex after v on its outline; (vx[v], vy[v]): the corner of v.
inline void LinkVertices(const VertexLists& lists, DArray<int32_t>* nxt, DArray<int32_t>* vx,
                         DArray<int32_t>* vy) {
  const int64_t nv = lists.size();
  nxt->Reset(nv);
  vx->Reset(nv);
  vy->Reset(nv);
  auto L = lists.DeviceView();
  auto n = nxt->DeviceView();
  auto ox = vx->DeviceView();
  auto oy = vy->DeviceView();
  ForEach(nv, [=] __device__(size_t v) {
    int64_t x, y;
    L.Corner(static_cast<int64_t>(v), &x, &y);
    ox[v] = static_cast<int32_t>(x);
    oy[v] = static_cast<int32_t>(y);
    n[v] = static_cast<int32_t>(L.Next(static_cast<int64_t>(v)));
  });
}

}  // namespace xrasp
#endif  // XRASP_STEPS_S2_LINKS_BASE_H
