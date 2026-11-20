#ifndef XRASP_STEPS_S7_ASSEMBLE_OPT_SCAN_ORDER_H
#define XRASP_STEPS_S7_ASSEMBLE_OPT_SCAN_ORDER_H
// Step 7, optional: the polygons in the order in which GDAL's polygonizer writes them, so that the output is
// the same sequence of features, not only the same set.
//
// GDAL scans the raster row by row and writes a polygon when the row after its last cell has been read.  Its
// order has three parts.
//
//   1. A ring starts at its top-left vertex (the smallest y, then the smallest x), and the holes of a polygon
//      follow in the order of those vertices.  Both are read off the rings.
//   2. Polygons are ordered by the row of their last cell: the largest y of the outer ring.
//   3. Polygons that end in the same row are ordered by a number GDAL gave them in a first scan.  That scan
//      numbers the cells that have neither their left nor their upper neighbour in their block (a "fresh"
//      cell starts a fragment), lets every other cell take the fragment of its left neighbour, or else of
//      its upper one, and merges two fragments when a cell finds its upper neighbour in another fragment
//      than its left one: the merged fragment keeps the number that the left neighbour's fragment had.  The
//      polygon's number is the one that is left at the end.
//
// Part 3 is the history of a union-find, but all of it happens at vertices, and nearly all of it can be read
// without replaying the scan:
//
//   - A run of cells of one block in one row starts at a unit edge walked downwards (the block on its east).
//     A straight stretch of such edges, from a vertex to the next vertex entered moving down, is a "left
//     edge"; it is named by the vertex that ends it.  All runs that start on one left edge have one fragment.
//   - The fragment of a left edge is new when the cell above its first cell is not of the block (a fresh
//     cell).  Otherwise it is the fragment of the run above: of the left edge found by going west from that
//     cell (the search of step 6).  Following these links (pointer jumping) gives every left edge its fragment.
//   - Fragments can only merge at a vertex entered moving down where the outline turns west and the cell
//     below-right is of the block: the run above (which starts at that left edge) meets the run of the row
//     (found by going west).  These vertices, in scan order, are the vertices entered moving left, in their
//     own order, whose previous vertex is such a one.
//
// What is left of the scan is the merges of each polygon in order, on its own fragments only: one thread
// replays the merges of one polygon, the polygons are independent.
#include <cstdint>

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

#include "xrasp/steps/s5_rings/base.h"
#include "xrasp/structs/rings.h"
#include "xrasp/structs/vertex_lists.h"
#include "xrasp/util/algo.h"
#include "xrasp/util/dev_array.h"
#include "xrasp/util/dev_value.h"
#include "xrasp/backend/launcher.h"

namespace xrasp {

struct ScanOrder {
  // per ring
  DArray<int64_t> first;        // corner position (y * wc + x) of the ring's top-left vertex
  DArray<int32_t> first_rank;   // position of that vertex in the ring (Rings::rank)
  DArray<int64_t> last_row;     // largest y of the ring; of the outer ring: GDAL writes the polygon after
                                // reading that row
  // per outer ring (unset for holes): of its polygon.  Empty until FindPolygonNumbers.
  DArray<int64_t> seed;         // cell position (y * w + x) of the fresh cell whose number the polygon keeps
};

struct IsOne {
  DEV_HOST_INLINE bool operator()(uint8_t f) const { return f == 1; }
};

struct IsGroupStart {
  const int64_t* key;
  DEV_HOST_INLINE bool operator()(int64_t i) const { return i == 0 || key[i] != key[i - 1]; }
};

// The left edge on which the run of cell (cx, cy) starts.  e: the sorted vertical unit edges (cy * wc + x),
// row[c]: where row c starts in e; down: the sorted positions (x * hc + y) of the vertices entered moving
// down, col[c]: where column c starts in down.  Both searches stay in one row and one column (the reason is
// that of s2_links/opt_line_index.h).
struct LeftEdges {
  const int64_t* e;
  const int64_t* row;
  const int64_t* down;
  const int64_t* col;
  int64_t wc, hc;
  DEV_INLINE int32_t Of(int64_t cx, int64_t cy) const {
    const int64_t r0 = row[cy], rn = row[cy + 1] - r0;
    // the last edge at or west of the cell
    const int64_t at = r0 + UpperBound(e + r0, rn, cy * wc + cx) - 1;
    const int64_t xh = e[at] % wc;
    const int64_t c0 = col[xh], cn = col[xh + 1] - c0;
    return static_cast<int32_t>(c0 + LowerBound(down + c0, cn, xh * hc + (cy + 1)));
  }
};

// Parts 1 and 2 of the order: the top-left vertex and the largest y of every ring.  wc: corners in a row.
// vx, vy: the corner of every vertex.  This is all that is needed to put the rings and the vertices of every
// polygon in GDAL's order, and the polygons in the order of their last row.
inline void FindRingStarts(int64_t wc, const DArray<int32_t>& vx, const DArray<int32_t>& vy, const Rings& rings,
                           ScanOrder* order) {
  const int64_t nv = rings.nv, nr = rings.nr;
  order->first.Reset(nr);
  order->first_rank.Reset(nr);
  order->last_row.Reset(nr);
  DArray<int64_t> low(nr);                // -(largest y)
  {
    auto first = order->first.DeviceView();
    auto lw = low.DeviceView();
    ForEach(nr, [=] __device__(size_t r) {
      first[r] = INT64_MAX;
      lw[r] = 0;
    });
    auto x = vx.DeviceView(), y = vy.DeviceView();
    auto ring = rings.ring.DeviceView(), rank = rings.rank.DeviceView();
    ForEach(nv, [=] __device__(size_t v) {
      AtomicMin64(&first[ring[v]], static_cast<int64_t>(y[v]) * wc + x[v]);
      AtomicMin64(&lw[ring[v]], -static_cast<int64_t>(y[v]));
    });
    auto fr = order->first_rank.DeviceView();
    ForEach(nv, [=] __device__(size_t v) {
      if (static_cast<int64_t>(y[v]) * wc + x[v] == first[ring[v]]) fr[ring[v]] = rank[v];
    });
    auto last_row = order->last_row.DeviceView();
    ForEach(nr, [=] __device__(size_t r) { last_row[r] = -lw[r]; });
  }
}

// Part 3 of the order, what it needs of the links: top_y[u], the y of the vertex at which the left edge that
// ends at u starts (its x is that of u), and from[j], the vertex before the j-th vertex entered moving left.
// nxt, vy: next vertex and y of every vertex, after the diagonal repair.  After this the links and the
// corners of the vertices are not read any more.
inline void FindEdgeTops(const VertexLists& lists, const DArray<int32_t>& nxt, const DArray<int32_t>& vy,
                         DArray<int32_t>* top_y, DArray<int32_t>* from) {
  const int64_t nv = static_cast<int64_t>(nxt.size());
  const int64_t n0 = lists.count[0], n3 = lists.count[3], base3 = lists.base[3];
  top_y->Reset(n0);
  from->Reset(n3);
  auto vn = nxt.DeviceView(), y = vy.DeviceView();
  auto t = top_y->DeviceView(), f = from->DeviceView();
  ForEach(nv, [=] __device__(size_t v) {
    int64_t u = vn[v];
    if (u < n0) {
      t[u] = y[v];
    } else if (u >= base3) {
      f[u - base3] = static_cast<int32_t>(v);
    }
  });
}

// Part 3 of the order: the number GDAL's first scan leaves on every polygon (order->seed, at its outer ring).
// cells: the raster (h rows of w cells).  top_y, from: FindEdgeTops.  rings: parent (step 6), and the ring of
// the vertices entered moving down (the first lists.count[0] vertices); their corners are read from the list.
template <typename T>
void FindPolygonNumbers(const T* cells, int64_t w, const VertexLists& lists, const DArray<int64_t>& edges,
                        const DArray<int32_t>& top_y, const DArray<int32_t>& from, const Rings& rings,
                        ScanOrder* order) {
  const int64_t nr = rings.nr;
  const int64_t wc = lists.wc, hc = lists.hc;
  const int64_t n0 = lists.count[0], n3 = lists.count[3];
  const int64_t ne = static_cast<int64_t>(edges.size());
  const int64_t* down = lists.pos[0].data();        // x * hc + y of the vertices entered moving down
  // where every row starts in the edges, and every column in the vertices entered moving down
  DArray<int64_t> row_start(hc), col_start(wc + 1);
  {
    auto rs = row_start.DeviceView(), cs = col_start.DeviceView();
    const int64_t* e = edges.data();
    ForEach(hc, [=] __device__(size_t c) { rs[c] = LowerBound(e, ne, static_cast<int64_t>(c) * wc); });
    ForEach(wc + 1, [=] __device__(size_t c) { cs[c] = LowerBound(down, n0, static_cast<int64_t>(c) * hc); });
  }
  const LeftEdges left{edges.data(), row_start.data(), lists.pos[0].data(), col_start.data(), wc, hc};
  // up[u]: u itself when the first cell of the left edge is fresh, the left edge of the run above otherwise
  DArray<int32_t> up(n0);
  {
    auto t = top_y.DeviceView();
    auto p = up.DeviceView();
    ForEach(n0, [=] __device__(size_t u) {
      int64_t cx = down[u] / hc, cy = t[u];        // the first cell of the left edge
      bool fresh = cy == 0 || cells[(cy - 1) * w + cx] != cells[cy * w + cx];
      p[u] = fresh ? static_cast<int32_t>(u) : left.Of(cx, cy - 1);
    });
  }
  // left edge -> ... -> the left edge of its fragment
  {
    DArray<int32_t> next(n0);
    DValue<int> changed;
    while (true) {
      changed.SetH2D(0);
      auto p = up.DeviceView();
      auto q = next.DeviceView();
      auto flag = changed.DeviceView();
      ForEach(n0, [=] __device__(size_t u) {
        int32_t pp = p[p[u]];
        q[u] = pp;
        if (pp != p[u]) flag.Set(1);
      });
      if (changed.GetD2H() == 0) break;
      up.Swap(next);
    }
  }

  // the merges, in scan order: packed (fragment of the run above) << 32 | (fragment of the run of the row)
  DArray<uint8_t> is_merge(n3);
  {
    auto f = from.DeviceView();
    auto m = is_merge.DeviceView();
    ForEach(n3, [=] __device__(size_t j) {
      int32_t c = f[j];                             // entered moving down, leaves moving west?
      if (c >= n0) {
        m[j] = 0;
        return;
      }
      int64_t cx = down[c] / hc, cy = down[c] % hc; // the cell below-right; the one west of it is of the block
      m[j] = cells[cy * w + cx] == cells[cy * w + cx - 1] ? 1 : 0;
    });
  }
  const int64_t nm = thrust::count_if(Exec(), is_merge.begin(), is_merge.end(), IsOne());
  DArray<int32_t> uf(n0);                           // union-find over the fragments (the left edges with up = self)
  {
    auto p = uf.DeviceView();
    ForEach(n0, [=] __device__(size_t u) { p[u] = static_cast<int32_t>(u); });
  }
  if (nm > 0) {
    DArray<int64_t> sel(nm), poly(nm), pair(nm);
    thrust::copy_if(Exec(), thrust::make_counting_iterator<int64_t>(0), thrust::make_counting_iterator<int64_t>(n3),
                    is_merge.begin(), sel.begin(), IsOne());
    is_merge.Free();
    {
      auto f = from.DeviceView();
      auto frag = up.DeviceView();
      auto ring = rings.ring.DeviceView(), parent = rings.parent.DeviceView();
      auto s = sel.DeviceView(), po = poly.DeviceView(), pr = pair.DeviceView();
      ForEach(nm, [=] __device__(size_t k) {
        int32_t c = f[s[k]];
        int64_t above = frag[c];
        int64_t here = frag[left.Of(down[c] / hc, down[c] % hc)];
        po[k] = parent[ring[c]];
        pr[k] = (above << 32) | here;
      });
    }
    sel.Free();
    // polygon by polygon, each still in scan order
    thrust::stable_sort_by_key(Exec(), poly.begin(), poly.end(), pair.begin());
    DArray<int64_t> start_all(nm);
    int64_t* start_end = thrust::copy_if(Exec(), thrust::make_counting_iterator<int64_t>(0),
                                         thrust::make_counting_iterator<int64_t>(nm), start_all.begin(),
                                         IsGroupStart{poly.data()});
    const int64_t ng = start_end - start_all.begin();
    {
      auto st = start_all.DeviceView();
      auto pr = pair.DeviceView();
      auto p = uf.DeviceView();
      ForEach(ng, [=] __device__(size_t g) {
        const int64_t last = static_cast<int64_t>(g) + 1 < ng ? st[g + 1] : nm;
        for (int64_t k = st[g]; k < last; k++) {
          int32_t a = static_cast<int32_t>(pr[k] >> 32), b = static_cast<int32_t>(pr[k] & 0xFFFFFFFFll);
          while (p[a] != a) {
            p[a] = p[p[a]];
            a = p[a];
          }
          while (p[b] != b) {
            p[b] = p[p[b]];
            b = p[b];
          }
          if (a != b) p[a] = b;                     // the number of the row's run stays
        }
      });
    }
  } else {
    is_merge.Free();
  }
  // one fragment of every ring; the polygon's number is that of the fragment its outer ring's one ended in
  DArray<int64_t> any(nr);
  order->seed.Reset(nr);
  {
    auto a = any.DeviceView();
    ForEach(nr, [=] __device__(size_t r) { a[r] = INT64_MAX; });
    auto frag = up.DeviceView();
    auto ring = rings.ring.DeviceView();
    ForEach(n0, [=] __device__(size_t u) { AtomicMin64(&a[ring[u]], frag[u]); });
    auto p = uf.DeviceView();
    auto t = top_y.DeviceView();
    auto parent = rings.parent.DeviceView();
    auto seed = order->seed.DeviceView();
    ForEach(nr, [=] __device__(size_t r) {
      if (parent[r] != static_cast<int32_t>(r)) return;
      int32_t f = static_cast<int32_t>(a[r]);
      while (p[f] != f) f = p[f];
      seed[r] = static_cast<int64_t>(t[f]) * w + down[f] / hc;
    });
  }
}

// The three parts.
template <typename T>
void FindScanOrder(const T* cells, int64_t w, const VertexLists& lists, const DArray<int64_t>& edges,
                   const DArray<int32_t>& nxt, const DArray<int32_t>& vx, const DArray<int32_t>& vy,
                   const Rings& rings, ScanOrder* order) {
  FindRingStarts(lists.wc, vx, vy, rings, order);
  DArray<int32_t> top_y, from;
  FindEdgeTops(lists, nxt, vy, &top_y, &from);
  FindPolygonNumbers(cells, w, lists, edges, top_y, from, rings, order);
}

}  // namespace xrasp
#endif  // XRASP_STEPS_S7_ASSEMBLE_OPT_SCAN_ORDER_H
