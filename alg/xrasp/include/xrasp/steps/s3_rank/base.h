#ifndef XRASP_STEPS_S3_RANK_BASE_H
#define XRASP_STEPS_S3_RANK_BASE_H
// Step 3: the ring of every vertex and its place in the ring.
//
// Pointer jumping over all vertices together.  After k rounds every vertex knows the smallest vertex number
// among the next 2^k vertices of its ring (m), how many steps ahead that vertex is (off), and the vertex 2^k
// steps ahead (j).  It stops when a round changes nothing: m is then the ring's smallest vertex -- the name
// of the ring -- and off the distance to it.  The number of rounds is the logarithm of the longest ring.
#include <cstdint>
#include <utility>

#include <thrust/sequence.h>

#include "xrasp/util/dev_array.h"
#include "xrasp/util/dev_value.h"
#include "xrasp/backend/launcher.h"

namespace xrasp {

// The loop itself.  j: where every vertex points; m: its key (consumed, both).  With keys other than the
// vertex numbers the lists may also be chains: a chain ends at a vertex that points to itself and whose key is
// smaller than every key before it on the chain (s3_rank/opt_two_level.h); a vertex on a chain then gets the
// key of that end and the number of steps to it.  Returns the number of rounds.
inline int RankJump(DArray<int32_t>* j_in, DArray<int32_t>* m_in, DArray<int32_t>* m_out, DArray<int32_t>* off_out) {
  const size_t nv = j_in->size();
  DArray<int32_t> m = std::move(*m_in), j = std::move(*j_in);
  DArray<int32_t> off(nv, 0), m2(nv), off2(nv), j2(nv);
  DValue<int> changed;
  int32_t step = 1;
  int rounds = 0;
  while (nv > 0) {
    changed.SetH2D(0);
    auto vm = m.DeviceView(), voff = off.DeviceView(), vj = j.DeviceView();
    auto wm = m2.DeviceView(), woff = off2.DeviceView(), wj = j2.DeviceView();
    auto flag = changed.DeviceView();
    ForEach(nv, [=] __device__(size_t i) {
      int32_t ahead = vj[i];
      int32_t mj = vm[ahead];
      if (mj < vm[i]) {
        wm[i] = mj;
        woff[i] = voff[ahead] + step;
        flag.Set(1);
      } else {
        wm[i] = vm[i];
        woff[i] = voff[i];
      }
      wj[i] = vj[ahead];
    });
    if (changed.GetD2H() == 0) break;
    m.Swap(m2);
    off.Swap(off2);
    j.Swap(j2);
    step *= 2;
    rounds++;
  }
  *m_out = std::move(m);
  *off_out = std::move(off);
  return rounds;
}

// All vertices together: every vertex points to its next vertex, its key is its own number.
inline int RankFlat(const DArray<int32_t>& nxt, DArray<int32_t>* m_out, DArray<int32_t>* off_out) {
  const size_t nv = nxt.size();
  DArray<int32_t> m(nv), j(nv);
  thrust::sequence(Exec(), m.begin(), m.end(), 0);
  if (nv > 0) {
    CopyOnDevice(j.data(), nxt.data(), nv * sizeof(int32_t));
  }
  return RankJump(&j, &m, m_out, off_out);
}

}  // namespace xrasp
#endif  // XRASP_STEPS_S3_RANK_BASE_H
