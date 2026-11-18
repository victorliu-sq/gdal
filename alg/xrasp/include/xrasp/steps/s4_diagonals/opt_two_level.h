#ifndef XRASP_STEPS_S4_DIAGONALS_OPT_TWO_LEVEL_H
#define XRASP_STEPS_S4_DIAGONALS_OPT_TWO_LEVEL_H
// Optimization of step 4, the second half of the two-level ranking (s3_rank/opt_two_level.h).
//
// What it changes.  The base swaps the successors of the two vertices of a fused diagonal corner and ranks
// ALL vertices again.  Here both vertices are ends of pieces, so the swap is a swap of the successors of two
// pieces and only level 2 is ranked again.  Then the result is written back to the vertices in one pass:
// the smallest vertex of the ring, and how many steps ahead it is -- the same numbers the base produces.
#include <cstdint>

#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include "xrasp/steps/s3_rank/opt_two_level.h"
#include "xrasp/steps/s4_diagonals/base.h"
#include "xrasp/util/dev_array.h"
#include "xrasp/util/launcher.h"

namespace xrasp {

struct SameRingOfPieces {
  const int32_t* a;
  const int32_t* b;
  const int32_t* cid;
  const int32_t* m2;
  DEV_HOST_INLINE int64_t operator()(int64_t k) const {
    return b[k] >= 0 && m2[cid[a[k]]] == m2[cid[b[k]]] ? 1 : 0;
  }
};

// m, off: as RankTwoLevel left them; final on return.  two: consumed.
inline DiagonalRepair RepairDiagonalsTwoLevel(const DArray<int32_t>& a, const DArray<int32_t>& b,
                                              DArray<int32_t>* nxt, DArray<int32_t>* m, DArray<int32_t>* off,
                                              TwoLevel* two) {
  DiagonalRepair r;
  const int64_t n = static_cast<int64_t>(a.size());
  const int64_t nv = static_cast<int64_t>(nxt->size());
  const int64_t pieces = two->pieces;
  r.pairs = thrust::count_if(Exec(), b.begin(), b.end(), IsPair());
  r.one_block = thrust::transform_reduce(
      Exec(), thrust::make_counting_iterator<int64_t>(0), thrust::make_counting_iterator<int64_t>(n),
      SameRingOfPieces{a.data(), b.data(), two->cid.data(), two->m2.data()}, static_cast<int64_t>(0),
      thrust::plus<int64_t>());
  if (r.one_block > 0) {
    auto va = a.DeviceView(), vb = b.DeviceView();
    auto cid = two->cid.DeviceView(), m2 = two->m2.DeviceView();
    auto vn = nxt->DeviceView(), n2 = two->nxt2.DeviceView();
    ForEach(n, [=] __device__(size_t k) {
      if (vb[k] < 0) return;
      int32_t pa = cid[va[k]], pb = cid[vb[k]];
      if (m2[pa] != m2[pb]) return;
      int32_t t = vn[va[k]];                 // the vertices' successors ...
      vn[va[k]] = vn[vb[k]];
      vn[vb[k]] = t;
      t = n2[pa];                            // ... and the pieces'.  The pieces themselves stay as they are
      n2[pa] = n2[pb];
      n2[pb] = t;
    });
    two->m2.Free();
    two->off2.Free();
    r.rounds = RankPieces(two->nxt2, two->pmin, two->plen, &two->m2, &two->off2);
  }
  two->cid.Free();
  two->pmin.Free();

  // back to the vertices.  Per piece first: the length of its ring, and the number to add to a vertex's
  // "steps to the end of my piece" to get its "steps to the ring's smallest vertex"
  DArray<int32_t> ring_total(pieces), shift(pieces);
  {
    auto m2 = two->m2.DeviceView(), n2 = two->nxt2.DeviceView(), len = two->plen.DeviceView();
    auto off2 = two->off2.DeviceView();
    auto vm = m->DeviceView(), voff = off->DeviceView();
    auto total = ring_total.DeviceView(), sh = shift.DeviceView();
    const int32_t np = static_cast<int32_t>(pieces);
    ForEach(pieces, [=] __device__(size_t i) {
      int32_t head = m2[i];                              // the ring's smallest vertex
      int32_t first = vm[head] + np;                     // the piece that holds it
      total[i] = static_cast<int32_t>(off2[n2[first]] + len[first]);
      sh[i] = static_cast<int32_t>(off2[i] - len[i] + len[first] - voff[head]);
    });
  }
  {
    auto vm = m->DeviceView(), voff = off->DeviceView();
    auto m2 = two->m2.DeviceView();
    auto total = ring_total.DeviceView(), sh = shift.DeviceView();
    const int32_t np = static_cast<int32_t>(pieces);
    ForEach(nv, [=] __device__(size_t v) {
      if (vm[v] >= 0) return;                            // on an uncut ring: already final
      int32_t node = vm[v] + np;
      int32_t o = voff[v] + sh[node];
      voff[v] = o < 0 ? o + total[node] : o;
      vm[v] = m2[node];
    });
  }
  *two = TwoLevel();
  return r;
}

}  // namespace xrasp
#endif  // XRASP_STEPS_S4_DIAGONALS_OPT_TWO_LEVEL_H
