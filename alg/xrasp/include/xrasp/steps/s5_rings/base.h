#ifndef XRASP_STEPS_S5_RINGS_BASE_H
#define XRASP_STEPS_S5_RINGS_BASE_H
// Step 5: from "smallest vertex of my ring and distance to it" to rings.
//
// Rings are numbered in the order of their smallest vertex (the head).  A ring is an outer ring or a hole by
// the sign of its area: with y pointing down and the block on the left of the outline, an outer ring comes
// out negative and a hole positive.  Every ring also gets one cell of its own block (next to its head), and
// every hole the cell just west of the top end of its leftmost edge -- a cell of the block the hole is in,
// from which step 6 looks for the ring that surrounds it.
#include <cstdint>

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

#include "xrasp/structs/rings.h"
#include "xrasp/structs/vertex_lists.h"
#include "xrasp/util/dev_array.h"
#include "xrasp/backend/launcher.h"

namespace xrasp {

DEV_INLINE void AtomicMin64(int64_t* address, int64_t value) {
  unsigned long long* a = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *a;
  while (value < static_cast<int64_t>(old)) {
    unsigned long long seen = atomicCAS(a, old, static_cast<unsigned long long>(value));
    if (seen == old) break;
    old = seen;
  }
}

DEV_INLINE void AtomicAdd64(int64_t* address, int64_t value) {
  atomicAdd(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(value));
}

struct IsHead {
  const int32_t* m;
  DEV_HOST_INLINE bool operator()(int32_t v) const { return m[v] == v; }
};

struct IsSet {
  DEV_HOST_INLINE bool operator()(uint8_t f) const { return f != 0; }
};

// m[v]: smallest vertex of v's ring; off[v]: steps from v to it.  m and off are consumed.
inline void BuildRings(const VertexLists& lists, const DArray<int32_t>& nxt, const DArray<int32_t>& vx,
                       const DArray<int32_t>& vy, DArray<int32_t>* m, DArray<int32_t>* off, Rings* rings) {
  const int64_t nv = static_cast<int64_t>(nxt.size());
  const int64_t hc = lists.hc;
  rings->nv = nv;
  // the heads, in increasing order: head[r] is the smallest vertex of ring r
  DArray<int32_t> head_all(nv);
  int32_t* head_end = thrust::copy_if(Exec(), thrust::make_counting_iterator<int32_t>(0),
                                      thrust::make_counting_iterator<int32_t>(static_cast<int32_t>(nv)),
                                      head_all.begin(), IsHead{m->data()});
  const int64_t nr = head_end - head_all.begin();
  head_all.Shrink(nr);
  DArray<int32_t> head = std::move(head_all);
  rings->nr = nr;

  // ring number of every vertex: (number of heads up to and including my head) - 1
  rings->ring.Reset(nv);
  {
    auto vm = m->DeviceView();
    auto r = rings->ring.DeviceView();
    ForEach(nv, [=] __device__(size_t v) { r[v] = vm[v] == static_cast<int32_t>(v) ? 1 : 0; });
    thrust::inclusive_scan(Exec(), rings->ring.begin(), rings->ring.end(), rings->ring.begin());
    // a non-head reads the count at its head; heads are read by others, so write into a second array
    DArray<int32_t> ring2(nv);
    auto r2 = ring2.DeviceView();
    ForEach(nv, [=] __device__(size_t v) { r2[v] = r[vm[v]] - 1; });
    rings->ring.Swap(ring2);
  }
  // length of every ring: the vertex after the head is length - 1 steps away from the head
  rings->length.Reset(nr);
  {
    auto h = head.DeviceView();
    auto voff = off->DeviceView();
    auto vn = nxt.DeviceView();
    auto len = rings->length.DeviceView();
    ForEach(nr, [=] __device__(size_t r) { len[r] = voff[vn[h[r]]] + 1; });
  }
  // position in the ring, 0 at the head, counting along the outline
  rings->rank.Reset(nv);
  {
    auto voff = off->DeviceView();
    auto r = rings->ring.DeviceView();
    auto len = rings->length.DeviceView();
    auto rk = rings->rank.DeviceView();
    ForEach(nv, [=] __device__(size_t v) {
      int32_t n = len[r[v]];
      rk[v] = (n - voff[v]) % n;
    });
  }
  m->Free();
  off->Free();

  // twice the signed area, exact in 64-bit integers
  DArray<int64_t> area2(nr, 0);
  {
    auto x = vx.DeviceView(), y = vy.DeviceView(), vn = nxt.DeviceView();
    auto r = rings->ring.DeviceView();
    auto a = area2.DeviceView();
    ForEach(nv, [=] __device__(size_t v) {
      int32_t n = vn[v];
      AtomicAdd64(&a[r[v]], static_cast<int64_t>(x[v]) * y[n] - static_cast<int64_t>(x[n]) * y[v]);
    });
  }
  rings->is_hole.Reset(nr);
  rings->cell_x.Reset(nr);
  rings->cell_y.Reset(nr);
  {
    auto a = area2.DeviceView();
    auto hole = rings->is_hole.DeviceView();
    auto h = head.DeviceView();
    auto x = vx.DeviceView(), y = vy.DeviceView();
    auto cx = rings->cell_x.DeviceView(), cy = rings->cell_y.DeviceView();
    auto L = lists.DeviceView();
    ForEach(nr, [=] __device__(size_t r) {
      hole[r] = a[r] > 0 ? 1 : 0;
      int d = L.Dir(h[r]);                       // the block is on the left of the direction of travel
      cx[r] = x[h[r]] - ((d == 1 || d == 2) ? 1 : 0);
      cy[r] = y[h[r]] - ((d == 0 || d == 1) ? 1 : 0);
    });
  }
  area2.Free();

  // the holes
  DArray<int32_t> holes_all(nr);
  int32_t* holes_end = thrust::copy_if(Exec(), thrust::make_counting_iterator<int32_t>(0),
                                       thrust::make_counting_iterator<int32_t>(static_cast<int32_t>(nr)),
                                       rings->is_hole.begin(), holes_all.begin(), IsSet());
  const int64_t nholes = holes_end - holes_all.begin();
  holes_all.Shrink(nholes);
  rings->holes = std::move(holes_all);
  rings->nholes = nholes;
  rings->hole_cx.Reset(nholes);
  rings->hole_cy.Reset(nholes);
  if (nholes > 0) {
    // smallest column-major position among the ring's vertices: the top end of its leftmost edge
    DArray<int64_t> kmin(nr, lists.wc * lists.hc);
    auto x = vx.DeviceView(), y = vy.DeviceView();
    auto r = rings->ring.DeviceView();
    auto k = kmin.DeviceView();
    ForEach(nv, [=] __device__(size_t v) { AtomicMin64(&k[r[v]], static_cast<int64_t>(x[v]) * hc + y[v]); });
    auto hs = rings->holes.DeviceView();
    auto cx = rings->hole_cx.DeviceView(), cy = rings->hole_cy.DeviceView();
    ForEach(nholes, [=] __device__(size_t i) {
      int64_t key = k[hs[i]];
      cx[i] = key / hc - 1;
      cy[i] = key % hc;
    });
  }
}

}  // namespace xrasp
#endif  // XRASP_STEPS_S5_RINGS_BASE_H
