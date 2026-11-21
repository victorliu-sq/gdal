#ifndef XRASP_STEPS_BANDS_B1_OPEN_VERTICES_H
#define XRASP_STEPS_BANDS_B1_OPEN_VERTICES_H
// Band step 1 (mode=bands): which vertices of a band are on outlines that leave it?
//
// A band holds the vertices of some corner rows.  Link every vertex to its next vertex inside the band.  A
// horizontal run stays in its row, so its end is in the band; a vertical run that finds no vertex in its
// column inside the band leaves it.  A vertex is OPEN when following the links reaches such a run (pointer
// jumping with a logical or); the others are on rings that lie inside the band.
//
// The band is then split: the open vertices are set aside (packed, to host memory), the others form lists of
// their own.  Those lists are made of whole rings, so steps 2 to 5 run on them unchanged
// (core/ring_pipeline.h); the set-aside vertices of all bands go through the same steps once at the end.
#include <cstdint>
#include <utility>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

#include "xrasp/structs/vertex_lists.h"
#include "xrasp/util/dev_array.h"
#include "xrasp/backend/launcher.h"

namespace xrasp {

struct IsOne {
  DEV_HOST_INLINE bool operator()(uint8_t v) const { return v != 0; }
};
struct IsZero {
  DEV_HOST_INLINE bool operator()(uint8_t v) const { return v == 0; }
};

// j[v]: the vertex after v inside the band, or v itself when the run that leaves v leaves the band;
// leaves[v] = 1 in that case.
inline void LinkInsideBand(const VertexLists& lists, DArray<int32_t>* j_out, DArray<uint8_t>* leaves) {
  const int64_t nv = lists.size();
  j_out->Reset(nv);
  leaves->Reset(nv);
  {
    auto L = lists.DeviceView();
    auto vj = j_out->DeviceView();
    auto vr = leaves->DeviceView();
    ForEach(nv, [=] __device__(size_t v) {
      int d = L.Dir(static_cast<int64_t>(v));
      int64_t i = static_cast<int64_t>(v) - L.base[d];
      int64_t x, y;
      L.Corner(static_cast<int64_t>(v), &x, &y);
      int dn = (L.flag[d][i] & kFlagRight) ? ((d + 3) & 3) : ((d + 1) & 3);
      int64_t q = (dn == 0 || dn == 2) ? x * L.hc + y : y * L.wc + x;
      // with the index of the lines (s2_links/opt_line_index.h): only among the vertices of my row or column
      int64_t lo = 0, n = L.count[dn];
      if (L.line_start[dn] != nullptr) {
        const int64_t line = (dn == 0 || dn == 2) ? x : y;
        lo = L.line_start[dn][line];
        n = L.line_start[dn][line + 1] - lo;
      }
      const int64_t* list = L.pos[dn] + lo;
      int64_t k = (dn <= 1) ? UpperBound(list, n, q) : LowerBound(list, n, q) - 1;
      bool ok = true;                                   // a horizontal run stays in its row
      if (dn == 0 || dn == 2) {                         // a vertical run: still in my column?
        ok = k >= 0 && k < n && list[k] / L.hc == x;
      }
      vj[v] = ok ? static_cast<int32_t>(L.base[dn] + lo + k) : static_cast<int32_t>(v);
      vr[v] = ok ? 0 : 1;
    });
  }
}

// The same answer as FindOpenVertices() below, by walking.  Inside the band an outline that leaves it is cut
// into paths: each starts at a vertex that no vertex of the band leads to (the outline comes in there) and ends
// at a vertex whose run leaves the band.  Every such path is walked once from its start, by one thread; the
// vertices of the rings inside the band are never visited.  Work: one pass over the vertices to find the
// starts, then as many steps as there are open vertices (a few per cent of the band), where pointer jumping
// passes over all the vertices twice in every round.
// links (may be null): receives j of LinkInsideBand(), for KeptLinks().
inline void FindOpenVerticesByWalk(const VertexLists& lists, DArray<uint8_t>* open,
                                   DArray<int32_t>* links = nullptr) {
  const int64_t nv = lists.size();
  DArray<int32_t> j;
  DArray<uint8_t> reach;
  LinkInsideBand(lists, &j, &reach);
  DArray<uint8_t> led_to(nv, 0);
  {
    auto vj = j.DeviceView();
    auto vl = led_to.DeviceView();
    // a vertex has one vertex before it at most: no two threads write the same byte
    ForEach(nv, [=] __device__(size_t v) {
      if (vj[v] != static_cast<int32_t>(v)) vl[vj[v]] = 1;
    });
  }
  {
    auto vj = j.DeviceView();
    auto vl = led_to.DeviceView();
    auto vr = reach.DeviceView();
    ForEach(nv, [=] __device__(size_t v) {
      if (vl[v]) return;
      int32_t u = static_cast<int32_t>(v);
      for (int64_t steps = 0; steps < nv; steps++) {      // a path ends where the outline leaves the band
        vr[u] = 1;
        const int32_t next = vj[u];
        if (next == u) break;
        u = next;
      }
    });
  }
  *open = std::move(reach);
  if (links != nullptr) *links = std::move(j);
}

// The band split in one pass over its vertices.  number[v]: how many vertices before v stay (one scan over
// the whole band); a vertex that stays goes to place number[v] of the kept vertices, an open one to place
// v - number[v] of the open ones -- each in its direction group, the groups in the same order as before.
// packed_open[d]: the open vertices of group d as position * 4 + flag (what is set aside); kept: the others,
// in order; owner: for every down-entered vertex, its index among the kept ones, or -1 when it is open.
struct SplitTargets {
  int64_t* open[4];
  int64_t* pos[4];
  uint8_t* flag[4];
  int64_t kept_base[4], open_base[4];
};

inline void SplitLists(const VertexLists& lists, const DArray<uint8_t>& open, DArray<int32_t>* number,
                       DArray<int64_t> packed_open[4], VertexLists* kept, DArray<int32_t>* owner) {
  const int64_t nv = lists.size();
  number->Reset(nv);
  {
    auto o = open.DeviceView();
    auto n = number->DeviceView();
    ForEach(nv, [=] __device__(size_t v) { n[v] = o[v] ? 0 : 1; });
  }
  const int64_t n_open = thrust::count_if(Exec(), open.begin(), open.end(), IsOne());
  thrust::exclusive_scan(Exec(), number->begin(), number->end(), number->begin());
  int64_t kept_base[5], open_base[5];
  for (int d = 0; d < 4; d++) {
    const int64_t b = lists.base[d];
    kept_base[d] = b < nv ? static_cast<int64_t>(number->Get(b)) : nv - n_open;
    open_base[d] = b - kept_base[d];
  }
  kept_base[4] = nv - n_open;
  open_base[4] = n_open;
  SplitTargets t;
  for (int d = 0; d < 4; d++) {
    packed_open[d].Reset(open_base[d + 1] - open_base[d]);
    kept->pos[d].Reset(kept_base[d + 1] - kept_base[d]);
    kept->flag[d].Reset(kept_base[d + 1] - kept_base[d]);
    t.open[d] = packed_open[d].data();
    t.pos[d] = kept->pos[d].data();
    t.flag[d] = kept->flag[d].data();
    t.kept_base[d] = kept_base[d];
    t.open_base[d] = open_base[d];
  }
  {
    auto L = lists.DeviceView();
    auto o = open.DeviceView();
    auto n = number->DeviceView();
    ForEach(nv, [=] __device__(size_t v) {
      const int d = L.Dir(static_cast<int64_t>(v));
      const int64_t i = static_cast<int64_t>(v) - L.base[d];
      if (o[v]) {
        t.open[d][static_cast<int64_t>(v) - n[v] - t.open_base[d]] = L.pos[d][i] * 4 + L.flag[d][i];
      } else {
        const int64_t k = n[v] - t.kept_base[d];
        t.pos[d][k] = L.pos[d][i];
        t.flag[d][k] = L.flag[d][i];
      }
    });
  }
  owner->Reset(lists.count[0]);
  {
    auto o = open.DeviceView();
    auto n = number->DeviceView();
    auto ow = owner->DeviceView();
    ForEach(lists.count[0], [=] __device__(size_t i) { ow[i] = o[i] ? -1 : n[i]; });
  }
}

// The vertices that are not open are whole rings, and the vertex after one of them inside the band is the
// vertex after it on its ring: their links are those of the band, in the numbers the kept vertices get
// (number of SplitLists).  Step 2 has nothing left to search for them.
inline void KeptLinks(const DArray<int32_t>& j, const DArray<uint8_t>& open, const DArray<int32_t>& number,
                      int64_t n_kept, DArray<int32_t>* nxt) {
  const int64_t nv = static_cast<int64_t>(j.size());
  nxt->Reset(n_kept);
  auto o = open.DeviceView();
  auto n = number.DeviceView();
  auto vj = j.DeviceView();
  auto out = nxt->DeviceView();
  ForEach(nv, [=] __device__(size_t v) {
    if (!o[v]) out[n[v]] = n[vj[v]];
  });
}

// open[v] = 1 for every vertex whose outline leaves the band.  walk: FindOpenVerticesByWalk(), which also
// gives the links of the band when links is not null (the jumps below overwrite theirs: links stays empty).
inline void FindOpenVertices(const VertexLists& lists, DArray<uint8_t>* open, bool walk = false,
                             DArray<int32_t>* links = nullptr) {
  if (walk) {
    FindOpenVerticesByWalk(lists, open, links);
    return;
  }
  const int64_t nv = lists.size();
  DArray<int32_t> j, j2(nv);
  DArray<uint8_t> reach, reach2(nv);
  LinkInsideBand(lists, &j, &reach);
  int64_t n = thrust::count_if(Exec(), reach.begin(), reach.end(), IsOne());
  while (n > 0) {                                       // after k rounds: a leaving run within 2^k - 1 steps
    auto vr = reach.DeviceView(), wr = reach2.DeviceView();
    auto vj = j.DeviceView();
    ForEach(nv, [=] __device__(size_t v) { wr[v] = vr[v] | vr[vj[v]]; });
    int64_t n2 = thrust::count_if(Exec(), reach2.begin(), reach2.end(), IsOne());
    reach.Swap(reach2);
    if (n2 == n) break;
    n = n2;
    auto wj = j2.DeviceView();
    ForEach(nv, [=] __device__(size_t v) { wj[v] = vj[vj[v]]; });
    j.Swap(j2);
  }
  *open = std::move(reach);
}

// Split one direction group.  packed_open: the open vertices as position * 4 + flag (what is set aside);
// kept_pos / kept_flag: the others, in order.  owner (may be null): for every vertex of the group, its index
// among the kept ones, or -1 when it is open.
inline void SplitGroup(const DArray<int64_t>& pos, const DArray<uint8_t>& flag, const uint8_t* open,
                       DArray<int64_t>* packed_open, DArray<int64_t>* kept_pos, DArray<uint8_t>* kept_flag,
                       DArray<int32_t>* owner) {
  const int64_t n = static_cast<int64_t>(pos.size());
  const int64_t n_open = thrust::count_if(Exec(), open, open + n, IsOne());
  packed_open->Reset(n_open);
  kept_pos->Reset(n - n_open);
  kept_flag->Reset(n - n_open);
  DArray<int64_t> packed(n);
  {
    auto p = pos.DeviceView();
    auto f = flag.DeviceView();
    auto out = packed.DeviceView();
    ForEach(n, [=] __device__(size_t i) { out[i] = p[i] * 4 + f[i]; });
  }
  thrust::copy_if(Exec(), packed.begin(), packed.end(), open, packed_open->begin(), IsOne());
  packed.Free();
  thrust::copy_if(Exec(), pos.begin(), pos.end(), open, kept_pos->begin(), IsZero());
  thrust::copy_if(Exec(), flag.begin(), flag.end(), open, kept_flag->begin(), IsZero());
  if (owner != nullptr) {
    owner->Reset(n);
    auto o = owner->DeviceView();
    ForEach(n, [=] __device__(size_t i) { o[i] = open[i] ? 0 : 1; });
    thrust::exclusive_scan(Exec(), owner->begin(), owner->end(), owner->begin());
    ForEach(n, [=] __device__(size_t i) {
      if (open[i]) o[i] = -1;
    });
  }
}

}  // namespace xrasp
#endif  // XRASP_STEPS_BANDS_B1_OPEN_VERTICES_H
