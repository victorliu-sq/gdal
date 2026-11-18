#ifndef XRASP_STRUCTS_VERTEX_LISTS_H
#define XRASP_STRUCTS_VERTEX_LISTS_H
// The vertices (turning points) of all outlines, grouped by the direction in which the outline ENTERS them:
//
//   d = 0 moving down    d = 1 moving right    d = 2 moving up    d = 3 moving left
//
// Each group is a sorted list of positions.  For the horizontal directions (1, 3) the position of corner
// (x, y) is y * wc + x (row by row), for the vertical ones (0, 2) it is x * hc + y (column by column), so the
// vertices of one row (column) are neighbours in their list.  wc = w + 1 and hc = h + 1 count corners.
//
// A vertex is numbered base[d] + (its index in group d).  An outline that leaves a vertex in direction dn
// arrives at a vertex entered in direction dn: the nearest one ahead on the same row or column, which is the
// neighbour of the leaving vertex's own position in the list of group dn.  Next() finds it with one binary
// search, for every vertex independently.
#include <cstdint>

#include "xrasp/util/algo.h"
#include "xrasp/util/dev_array.h"

namespace xrasp {

constexpr uint8_t kFlagDiagonal = 1;   // left turn where the front-right cell has my value: decided after ranking
constexpr uint8_t kFlagRight = 2;      // right turn (otherwise left)

struct VertexListsView {
  const int64_t* pos[4];
  const uint8_t* flag[4];
  int64_t count[4];
  int64_t base[4];
  int64_t wc, hc;
  // s2_links/opt_line_index.h: where every column (d = 0, 2) or row (d = 1, 3) starts in pos[d], or null
  const int64_t* line_start[4];

  DEV_INLINE int Dir(int64_t v) const {
    return v >= base[2] ? (v >= base[3] ? 3 : 2) : (v >= base[1] ? 1 : 0);
  }
  // corner (x, y) of vertex v
  DEV_INLINE void Corner(int64_t v, int64_t* x, int64_t* y) const {
    int d = Dir(v);
    int64_t p = pos[d][v - base[d]];
    if (d == 0 || d == 2) {
      *x = p / hc;
      *y = p % hc;
    } else {
      *y = p / wc;
      *x = p % wc;
    }
  }
  // the vertex the outline reaches after leaving v
  DEV_INLINE int64_t Next(int64_t v) const {
    int d = Dir(v);
    int64_t i = v - base[d];
    int64_t x, y;
    Corner(v, &x, &y);
    int dn = (flag[d][i] & kFlagRight) ? ((d + 3) & 3) : ((d + 1) & 3);
    // my own position in the ordering of the group I am heading for
    int64_t q = (dn == 0 || dn == 2) ? x * hc + y : y * wc + x;
    // the next vertex is on my column (dn = 0, 2) or row: with the index, search only that part of the list
    int64_t lo = 0, n = count[dn];
    if (line_start[dn] != nullptr) {
      const int64_t line = (dn == 0 || dn == 2) ? x : y;
      lo = line_start[dn][line];
      n = line_start[dn][line + 1] - lo;
    }
    // moving down or right goes towards larger positions: the first entry after q; otherwise the last before q
    int64_t k = (dn <= 1) ? UpperBound(pos[dn] + lo, n, q) : LowerBound(pos[dn] + lo, n, q) - 1;
    return base[dn] + lo + k;
  }
};

struct VertexLists {
  DArray<int64_t> pos[4];
  DArray<uint8_t> flag[4];
  DArray<int64_t> line_start[4];           // empty unless s2_links/opt_line_index.h built it
  int64_t count[4] = {0, 0, 0, 0};
  int64_t base[4] = {0, 0, 0, 0};
  int64_t wc = 0, hc = 0;

  void SetCounts() {
    for (int d = 0; d < 4; d++) count[d] = static_cast<int64_t>(pos[d].size());
    base[0] = 0;
    for (int d = 1; d < 4; d++) base[d] = base[d - 1] + count[d - 1];
  }
  int64_t size() const { return count[0] + count[1] + count[2] + count[3]; }
  VertexListsView DeviceView() const {
    VertexListsView v;
    for (int d = 0; d < 4; d++) {
      v.pos[d] = pos[d].data();
      v.flag[d] = flag[d].data();
      v.count[d] = count[d];
      v.base[d] = base[d];
      v.line_start[d] = line_start[d].empty() ? nullptr : line_start[d].data();
    }
    v.wc = wc;
    v.hc = hc;
    return v;
  }
  // Keep only the positions of the vertices entered moving down (what steps 5 to 7 still read), and the
  // counts.
  void FreeAllButDown() {
    for (int d = 0; d < 4; d++) {
      if (d != 0) pos[d].Free();
      flag[d].Free();
      line_start[d].Free();
    }
  }
  void Free() {
    for (int d = 0; d < 4; d++) {
      pos[d].Free();
      flag[d].Free();
      line_start[d].Free();
    }
  }
};

}  // namespace xrasp
#endif  // XRASP_STRUCTS_VERTEX_LISTS_H
