#ifndef XRASP_STEPS_S3_RANK_OPT_TWO_LEVEL_H
#define XRASP_STEPS_S3_RANK_OPT_TWO_LEVEL_H
// Optimization of step 3 (and of step 4, see s4_diagonals/opt_two_level.h): rank in two levels.
//
// What it changes.  The base ranks all vertices together: the number of rounds is the logarithm of the longest
// ring of the raster, and after the diagonal repair everything is ranked a second time.  Here the link out of
// a vertex is CUT when the next vertex lies in another tile (tiles of `tile` x `tile` corners), and at both
// vertices of every diagonal corner.  That leaves
//
//   * rings without a cut (inside one tile, no diagonal corner): ranked as before, they are done;
//   * pieces: chains of vertices that end at a cut vertex.
//
// Level 1 ranks inside the pieces, so the rounds depend on the tile size, not on the raster.  Level 2 has one
// node per piece (a few per cent of the vertices): the pieces are ranked along their rings, weighted by their
// lengths.  The diagonal repair then only swaps the successors of two pieces and ranks level 2 again; the
// vertices are never ranked twice.  The result written back to the vertices (in step 4) is exactly the base's.
//
// Measured with the Python reference (whole WorldCover tile, RTX 3090): ranking + repair 543 -> 228 ms.
//
// Level 1 by walking (walk = true).  Pointer jumping makes every vertex take part in about ten rounds, each
// a read somewhere else in memory; on a CPU that is what the step costs.  A piece is short (tens of vertices),
// so one thread can walk it from its first vertex to its cut vertex: once to find the end and the length,
// once to write.  The pieces are walked in parallel, every vertex is touched by one thread only.  The rings
// the cuts leave whole lie inside one tile.  They are found first (a walk from the first vertex of every
// piece marks the vertices that are on pieces), and each is walked from its vertices entered moving down --
// every ring has one, and the smallest vertex of a ring is one of them: a walk stops at the first vertex
// smaller than the one it started from, so only the walk from the smallest vertex comes round, and that one
// writes the ring.  Level 2 and the result are the same; nothing is cut that the jumping version does not cut.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

#include "xrasp/steps/s3_rank/base.h"
#include "xrasp/structs/vertex_lists.h"
#include "xrasp/util/dev_array.h"
#include "xrasp/util/dev_value.h"
#include "xrasp/backend/launcher.h"

namespace xrasp {

struct RankProfile {
  // where the time of step 3 goes (milliseconds): the cuts, level 1, the nodes of level 2, their ranking
  double cuts_ms = 0, level1_ms = 0, nodes_ms = 0, level2_ms = 0;
  // with XRASP_TILE_STATS set: the vertices per tile -- the most, the median and the 99th percentile over the
  // tiles that have any, and how many tiles (and the vertices in them) hold more than 6000 and 12000
  int64_t tile_max = 0, tile_p50 = 0, tile_p99 = 0, tiles_used = 0;
  int64_t tiles_over_6k = 0, vertices_over_6k = 0, tiles_over_12k = 0, vertices_over_12k = 0;
};

// What level 2 knows, kept from step 3 to step 4.
struct TwoLevel {
  int64_t pieces = 0;           // 0: there was no cut, the vertices were ranked the plain way
  DArray<int32_t> cid;          // per vertex: number of the piece that ends at it (meaningful at cut vertices)
  DArray<int32_t> nxt2;         // per piece: the piece that follows
  DArray<int32_t> plen;         // per piece: number of vertices
  DArray<int32_t> pmin;         // per piece: smallest vertex in it (nv when it has no down-entered vertex)
  DArray<int32_t> m2;           // per piece: smallest vertex of its ring
  DArray<int64_t> off2;         // per piece: vertices from its start to the start of the piece holding that vertex
  int rounds1 = 0, rounds2 = 0;
  int64_t walk_cuts = 0;        // walk: the cuts added to the rings that had none
  RankProfile prof;
};

// Pointer jumping over rings of pieces: like RankJump, but a step over a piece counts its length.
inline int RankPieces(const DArray<int32_t>& nxt2, const DArray<int32_t>& key, const DArray<int32_t>& length,
                      DArray<int32_t>* m_out, DArray<int64_t>* off_out) {
  const size_t n = nxt2.size();
  DArray<int32_t> m(n), j(n), m2(n), j2(n);
  DArray<int64_t> off(n, 0), span(n), off2(n), span2(n);
  CopyOnDevice(m.data(), key.data(), n * sizeof(int32_t));
  CopyOnDevice(j.data(), nxt2.data(), n * sizeof(int32_t));
  {
    auto len = length.DeviceView();
    auto sp = span.DeviceView();
    ForEach(n, [=] __device__(size_t i) { sp[i] = len[i]; });
  }
  DValue<int> changed;
  int rounds = 0;
  while (true) {
    changed.SetH2D(0);
    auto vm = m.DeviceView(), vj = j.DeviceView();
    auto voff = off.DeviceView(), vspan = span.DeviceView();
    auto wm = m2.DeviceView(), wj = j2.DeviceView();
    auto woff = off2.DeviceView(), wspan = span2.DeviceView();
    auto flag = changed.DeviceView();
    ForEach(n, [=] __device__(size_t i) {
      int32_t ahead = vj[i];
      int32_t mj = vm[ahead];
      if (mj < vm[i]) {
        wm[i] = mj;
        woff[i] = voff[ahead] + vspan[i];
        flag.Set(1);
      } else {
        wm[i] = vm[i];
        woff[i] = voff[i];
      }
      int64_t s = vspan[i] + vspan[ahead];          // only used while it is shorter than the ring
      wspan[i] = s < (int64_t{1} << 40) ? s : (int64_t{1} << 40);
      wj[i] = vj[ahead];
    });
    if (changed.GetD2H() == 0) break;
    m.Swap(m2);
    off.Swap(off2);
    span.Swap(span2);
    j.Swap(j2);
    rounds++;
  }
  *m_out = std::move(m);
  *off_out = std::move(off);
  return rounds;
}

struct IsCut {
  DEV_HOST_INLINE bool operator()(int32_t c) const { return c != 0; }
};

// Level 1, then the first ranking of level 2.  pair_a / pair_b: the diagonal pairs (b < 0: no partner).
// On return: for a vertex on a piece, m[v] + pieces is the number of its piece and off[v] the steps to the
// piece's end (m[v] < 0); for a vertex on an uncut ring, m[v] and off[v] are final.
inline void RankTwoLevel(const VertexLists& lists, const DArray<int32_t>& nxt, const DArray<int32_t>& vx,
                         const DArray<int32_t>& vy, const DArray<int32_t>& pair_a, const DArray<int32_t>& pair_b,
                         int tile, DArray<int32_t>* m, DArray<int32_t>* off, TwoLevel* two, bool walk = false) {
  const int64_t nv = static_cast<int64_t>(nxt.size());
  const int64_t tx = (lists.wc + tile - 1) / tile;
  if (std::getenv("XRASP_TILE_STATS") != nullptr) {
    const int64_t ty = (lists.hc + tile - 1) / tile;
    DArray<int32_t> per_tile(tx * ty, 0);
    {
      auto x = vx.DeviceView(), y = vy.DeviceView();
      auto c = per_tile.DeviceView();
      ForEach(nv, [=] __device__(size_t v) {
        atomicAdd(&c[static_cast<int64_t>(y[v] / tile) * tx + x[v] / tile], 1);
      });
    }
    std::vector<int32_t> h = per_tile.ToHost();
    h.erase(std::remove(h.begin(), h.end(), 0), h.end());
    std::sort(h.begin(), h.end());
    two->prof.tiles_used = static_cast<int64_t>(h.size());
    if (!h.empty()) {
      two->prof.tile_max = h.back();
      two->prof.tile_p50 = h[h.size() / 2];
      two->prof.tile_p99 = h[h.size() * 99 / 100];
      for (int32_t n : h) {
        if (n > 6000) {
          two->prof.tiles_over_6k++;
          two->prof.vertices_over_6k += n;
        }
        if (n > 12000) {
          two->prof.tiles_over_12k++;
          two->prof.vertices_over_12k += n;
        }
      }
    }
  }
  DeviceSync();
  auto clock = std::chrono::steady_clock::now();
  auto lap = [&clock]() {
    DeviceSync();
    auto now = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - clock).count();
    clock = now;
    return ms;
  };
  // 1 at a cut vertex
  two->cid.Reset(nv);
  {
    auto x = vx.DeviceView(), y = vy.DeviceView(), vn = nxt.DeviceView();
    auto c = two->cid.DeviceView();
    ForEach(nv, [=] __device__(size_t v) {
      int32_t n = vn[v];
      int64_t mine = static_cast<int64_t>(y[v] / tile) * tx + x[v] / tile;
      int64_t next = static_cast<int64_t>(y[n] / tile) * tx + x[n] / tile;
      c[v] = mine != next ? 1 : 0;
    });
    auto a = pair_a.DeviceView(), b = pair_b.DeviceView();
    ForEach(pair_a.size(), [=] __device__(size_t k) {
      if (b[k] >= 0) {
        c[a[k]] = 1;
        c[b[k]] = 1;
      }
    });
  }
  DArray<int32_t> tails_all(nv);
  int32_t* tails_end = thrust::copy_if(Exec(), thrust::make_counting_iterator<int32_t>(0),
                                       thrust::make_counting_iterator<int32_t>(static_cast<int32_t>(nv)),
                                       two->cid.begin(), tails_all.begin(), IsCut());
  const int64_t pieces = tails_end - tails_all.begin();
  two->pieces = pieces;
  if (pieces == 0) {                      // nothing was cut: the plain ranking
    two->cid.Free();
    two->rounds1 = RankFlat(nxt, m, off);
    return;
  }
  tails_all.Shrink(pieces);
  DArray<int32_t> tails = std::move(tails_all);
  two->prof.cuts_ms += lap();

  if (walk) {
    // level 1 by walking.  state[v]: 1 at a cut vertex, 2 at the other vertices of a piece, 0 on a ring
    // without a cut
    DArray<uint8_t> state_array(nv);
    m->Reset(nv);
    off->Reset(nv);
    auto state = state_array.DeviceView();
    auto cid = two->cid.DeviceView(), vn = nxt.DeviceView();
    auto t = tails.DeviceView();
    auto vm = m->DeviceView(), voff = off->DeviceView();
    const int32_t np = static_cast<int32_t>(pieces);
    ForEach(nv, [=] __device__(size_t v) { state[v] = cid[v] != 0 ? 1 : 0; });
    thrust::inclusive_scan(Exec(), two->cid.begin(), two->cid.end(), two->cid.begin());
    // the thread of piece i takes the piece that follows piece i, from its first vertex to its cut vertex
    ForEach(pieces, [=] __device__(size_t i) {
      const int32_t head = vn[t[i]];
      int32_t steps = 0, v = head;
      while (state[v] != 1) {
        state[v] = 2;
        v = vn[v];
        steps++;
      }
      const int32_t piece = cid[v] - 1 - np;         // the key of the piece: < 0
      v = head;
      while (true) {
        vm[v] = piece;
        voff[v] = steps;                              // steps to the piece's end
        if (steps == 0) break;
        steps--;
        v = vn[v];
      }
    });
    ForEach(nv, [=] __device__(size_t v) { cid[v] -= 1; });    // the piece that ends at the cut vertex at or before v
    // the rings without a cut, each from its smallest vertex
    ForEach(lists.count[0], [=] __device__(size_t v0) {
      const int32_t v = static_cast<int32_t>(v0);
      if (state[v] != 0) return;
      int32_t length = 1;
      for (int32_t u = vn[v]; u != v; u = vn[u]) {
        if (u < v) return;                            // not the smallest: another walk writes this ring
        length++;
      }
      vm[v] = v;
      voff[v] = 0;
      int32_t steps = length - 1;                     // steps from the vertex after v round to v
      for (int32_t u = vn[v]; u != v; u = vn[u]) {
        vm[u] = v;
        voff[u] = steps--;
      }
    });
    two->rounds1 = 0;
  } else {
    // level 1: a cut vertex points to itself and carries the key (its piece - pieces) < 0
    DArray<int32_t> key(nv), j0(nv);
    {
      DArray<int32_t> is_cut(nv);
      CopyOnDevice(is_cut.data(), two->cid.data(), nv * sizeof(int32_t));
      thrust::inclusive_scan(Exec(), two->cid.begin(), two->cid.end(), two->cid.begin());
      auto cut = is_cut.DeviceView(), cid = two->cid.DeviceView(), vn = nxt.DeviceView();
      auto k = key.DeviceView(), j = j0.DeviceView();
      const int32_t np = static_cast<int32_t>(pieces);
      ForEach(nv, [=] __device__(size_t v) {
        cid[v] -= 1;                                  // the piece that ends at the cut vertex at or before v
        k[v] = cut[v] ? cid[v] - np : static_cast<int32_t>(v);
        j[v] = cut[v] ? static_cast<int32_t>(v) : vn[v];
      });
    }
    two->rounds1 = RankJump(&j0, &key, m, off);
  }

  two->prof.level1_ms += lap();
  // level 2: one node per piece
  two->nxt2.Reset(pieces);
  two->plen.Reset(pieces);
  two->pmin.Reset(pieces);
  two->pmin.Fill(static_cast<int32_t>(nv));
  {
    auto t = tails.DeviceView(), vn = nxt.DeviceView();
    auto vm = m->DeviceView(), voff = off->DeviceView();
    auto n2 = two->nxt2.DeviceView(), len = two->plen.DeviceView(), pm = two->pmin.DeviceView();
    const int32_t np = static_cast<int32_t>(pieces);
    ForEach(pieces, [=] __device__(size_t i) {
      int32_t head = vn[t[i]];                      // first vertex of the piece that follows
      int32_t next = vm[head] + np;
      n2[i] = next;
      len[next] = voff[head] + 1;                   // every piece follows exactly one piece
    });
    // smallest vertex of every piece.  Every ring has a vertex entered moving down, and those have the
    // smallest numbers, so only they are looked at
    ForEach(lists.count[0], [=] __device__(size_t v) {
      if (vm[v] < 0) atomicMin(&pm[vm[v] + np], static_cast<int32_t>(v));
    });
  }
  two->prof.nodes_ms += lap();
  two->rounds2 = RankPieces(two->nxt2, two->pmin, two->plen, &two->m2, &two->off2);
  two->prof.level2_ms += lap();
}

}  // namespace xrasp
#endif  // XRASP_STEPS_S3_RANK_OPT_TWO_LEVEL_H
