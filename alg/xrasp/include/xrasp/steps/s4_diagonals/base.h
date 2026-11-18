#ifndef XRASP_STEPS_S4_DIAGONALS_BASE_H
#define XRASP_STEPS_S4_DIAGONALS_BASE_H
// Step 4: repair the corners where two cells of one value touch diagonally.
//
// Step 1 let both outlines turn around their own cell there, as if the two cells were different blocks.  If
// they are in fact one block (connected somewhere else), the two outlines that should pass through the corner
// have been fused into one ring.  That shows after ranking: the two vertices of that value at the corner
// (entered in opposite directions, 1 and 3 or 0 and 2) are in the same ring exactly when the two cells are
// one block.  Swapping their successors splits the ring correctly; then the vertices are ranked again.
//
// This is what replaces block labels: the rings themselves say which cells are connected.
#include <cstdint>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include "xrasp/steps/s3_rank/base.h"
#include "xrasp/structs/vertex_lists.h"
#include "xrasp/util/dev_array.h"
#include "xrasp/util/launcher.h"

namespace xrasp {

struct HasDiagonalFlag {
  DEV_HOST_INLINE bool operator()(uint8_t flag) const { return (flag & kFlagDiagonal) != 0; }
};

// The pairs (a[k], b[k]) of vertices of one value at a diagonal corner.  b[k] is -1 when the partner is not
// in the lists (they hold only part of the raster's vertices: the partner is then in another ring).
inline void FindDiagonalPairs(const VertexLists& lists, DArray<int32_t>* a, DArray<int32_t>* b) {
  const int from[2] = {1, 0}, partner[2] = {3, 2};
  int64_t counts[2];
  for (int g = 0; g < 2; g++) {
    counts[g] = thrust::count_if(Exec(), lists.flag[from[g]].begin(), lists.flag[from[g]].end(), HasDiagonalFlag());
  }
  a->Reset(counts[0] + counts[1]);
  b->Reset(counts[0] + counts[1]);
  auto L = lists.DeviceView();
  int64_t at = 0;
  for (int g = 0; g < 2; g++) {
    const int d = from[g], dp = partner[g];
    DArray<int64_t> sel(counts[g]);
    thrust::copy_if(Exec(), thrust::make_counting_iterator<int64_t>(0),
                    thrust::make_counting_iterator<int64_t>(lists.count[d]), lists.flag[d].begin(), sel.begin(),
                    HasDiagonalFlag());
    auto s = sel.DeviceView();
    auto va = a->DeviceView(), vb = b->DeviceView();
    ForEach(counts[g], [=] __device__(size_t k) {
      int64_t i = s[k];
      int64_t key = L.pos[d][i];
      int64_t kk = LowerBound(L.pos[dp], L.count[dp], key);          // the same corner in the opposite group
      bool here = kk < L.count[dp] && L.pos[dp][kk] == key;
      va[at + k] = static_cast<int32_t>(L.base[d] + i);
      vb[at + k] = here ? static_cast<int32_t>(L.base[dp] + kk) : -1;
    });
    at += counts[g];
  }
}

struct IsPair {
  DEV_HOST_INLINE bool operator()(int32_t b) const { return b >= 0; }
};

struct SameRing {
  const int32_t* a;
  const int32_t* b;
  const int32_t* m;
  DEV_HOST_INLINE int64_t operator()(int64_t k) const { return b[k] >= 0 && m[a[k]] == m[b[k]] ? 1 : 0; }
};

struct DiagonalRepair {
  int64_t pairs = 0;            // diagonal corners
  int64_t one_block = 0;        // of which the two cells turned out to be one block
  int rounds = 0;               // rounds of the second ranking (0: nothing had to be repaired)
};

// Swap the successors of the pairs that are in one ring, and rank again when there was any.
inline DiagonalRepair RepairDiagonalsFlat(const DArray<int32_t>& a, const DArray<int32_t>& b,
                                          DArray<int32_t>* nxt, DArray<int32_t>* m, DArray<int32_t>* off) {
  DiagonalRepair r;
  const int64_t n = static_cast<int64_t>(a.size());
  r.pairs = thrust::count_if(Exec(), b.begin(), b.end(), IsPair());
  r.one_block = thrust::transform_reduce(Exec(), thrust::make_counting_iterator<int64_t>(0),
                                         thrust::make_counting_iterator<int64_t>(n),
                                         SameRing{a.data(), b.data(), m->data()}, static_cast<int64_t>(0),
                                         thrust::plus<int64_t>());
  if (r.one_block == 0) return r;
  auto va = a.DeviceView(), vb = b.DeviceView();
  auto vm = m->DeviceView(), vn = nxt->DeviceView();
  ForEach(n, [=] __device__(size_t k) {
    int32_t x = va[k], y = vb[k];
    if (y >= 0 && vm[x] == vm[y]) {             // every vertex is in at most one pair: no two threads touch it
      int32_t t = vn[x];
      vn[x] = vn[y];
      vn[y] = t;
    }
  });
  m->Free();
  off->Free();
  r.rounds = RankFlat(*nxt, m, off);
  return r;
}

}  // namespace xrasp
#endif  // XRASP_STEPS_S4_DIAGONALS_BASE_H
