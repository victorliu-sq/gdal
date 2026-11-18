#ifndef XRASP_STEPS_S2_LINKS_OPT_LINE_INDEX_H
#define XRASP_STEPS_S2_LINKS_OPT_LINE_INDEX_H
// Optimization of step 2: search the next vertex only in its own row or column.
//
// The base searches the whole list of the direction a vertex leaves in: 26 probes in a list of 50 million
// positions, each probe somewhere else in memory.  The next vertex is on the same row (leaving right or left)
// or column (leaving down or up), and the vertices of one row or column are neighbours in their list.  With
// the start of every row or column in its list (one number each: a few hundred kilobytes), a vertex searches
// only among the vertices of its line -- about ten probes, all in a few kilobytes.  The vertex found is the
// same.
//
// On the CPU the probes of the base are cache misses and step 2 is the largest step on a dense raster
// (corine, 32 threads: 4.4 of 9.2 s).
#include <cstdint>

#include "xrasp/structs/vertex_lists.h"
#include "xrasp/util/algo.h"
#include "xrasp/util/dev_array.h"
#include "xrasp/util/launcher.h"

namespace xrasp {

// line_start[d][c]: the first entry of pos[d] on column c (d = 0, 2: positions x * hc + y) or row c
// (d = 1, 3: positions y * wc + x); one more entry at the end.
inline void BuildLineIndex(VertexLists* lists) {
  for (int d = 0; d < 4; d++) {
    const bool vertical = d == 0 || d == 2;
    const int64_t lines = vertical ? lists->wc : lists->hc;
    const int64_t stride = vertical ? lists->hc : lists->wc;
    lists->line_start[d].Reset(lines + 1);
    auto start = lists->line_start[d].DeviceView();
    const int64_t* pos = lists->pos[d].data();
    const int64_t n = lists->count[d];
    ForEach(lines + 1, [=] __device__(size_t c) {
      start[c] = LowerBound(pos, n, static_cast<int64_t>(c) * stride);
    });
  }
}

inline void FreeLineIndex(VertexLists* lists) {
  for (int d = 0; d < 4; d++) lists->line_start[d].Free();
}

}  // namespace xrasp
#endif  // XRASP_STEPS_S2_LINKS_OPT_LINE_INDEX_H
