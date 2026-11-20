#ifndef XRASP_STEPS_S1_CORNERS_BASE_H
#define XRASP_STEPS_S1_CORNERS_BASE_H
// Step 1: find the vertices.
//
// Around corner (x, y) are four cells:
//
//     UL = (x-1, y-1)   UR = (x, y-1)
//     LL = (x-1, y)     LR = (x, y)
//
// An outline that arrives at the corner moving in direction d has its own cell on the left (`own`), another
// value on the right (`other`), and ahead of it the cells `fl` (front left) and `fr` (front right).  It
//
//   goes straight on  when fl is mine and fr is not;
//   turns right       when fl and fr are both mine (the three cells are chained by shared edges: one block);
//   turns left        when fl is not mine.  If fr has my value it touches me only at this corner, and whether
//                     it is my block cannot be told from the four values: that is decided after ranking
//                     (step 4), and the vertex is flagged "diagonal".
//
// No block labels are needed: two cells that share an edge are one block exactly when their values are equal.
//
// One kernel computes, for every corner of a strip, a byte with two bits per direction (0 nothing or
// straight, 1 right, 2 left, 3 left at a diagonal corner).  The vertices of each direction are then the
// corners whose two bits are not 0, collected in order.  The same pass over the strip also collects the
// vertical unit edges (step 6 uses them to attach holes).
#include <cstdint>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

#include "xrasp/structs/dev_raster.h"
#include "xrasp/structs/vertex_lists.h"
#include "xrasp/util/dev_array.h"
#include "xrasp/backend/launcher.h"

namespace xrasp {

// The rule above, on raster values.  (s1_corners/with_labels.h has the rule on block labels.)
struct TurnByValue {
  static DEV_INLINE uint8_t State(int64_t own, int64_t other, int64_t fl, int64_t fr) {
    if (own == kOutside || own == other) return 0;       // no outline arrives here in this direction
    bool same_fl = fl == own, same_fr = fr == own;
    if (same_fl) return same_fr ? 1 : 0;                 // right turn : straight on
    return same_fr ? 3 : 2;                              // left turn, diagonal : plain
  }
};

// The byte of a corner from its four cells: two bits per direction.
template <typename Turn>
DEV_INLINE uint8_t CornerCodeOf(int64_t ul, int64_t ur, int64_t ll, int64_t lr) {
  return static_cast<uint8_t>(Turn::State(ur, ul, lr, ll)              // d = 0, moving down
                              | (Turn::State(ul, ll, ur, lr) << 2)     // d = 1, moving right
                              | (Turn::State(ll, lr, ul, ur) << 4)     // d = 2, moving up
                              | (Turn::State(lr, ur, ll, ul) << 6));   // d = 3, moving left
}

template <typename T, typename Turn>
DEV_INLINE uint8_t CornerCode(const DRasterView<T>& r, int64_t x, int64_t y) {
  return CornerCodeOf<Turn>(r.At(x - 1, y - 1), r.At(x, y - 1), r.At(x - 1, y), r.At(x, y));
}

struct IsVertexOf {
  int d;
  DEV_HOST_INLINE bool operator()(uint8_t code) const { return ((code >> (2 * d)) & 3) != 0; }
};

// Is there a vertical unit edge at position e = cy * wc + x: do cells (x-1, cy) and (x, cy) differ?
template <typename T>
struct IsVerticalEdge {
  DRasterView<T> raster;
  int64_t wc;
  DEV_INLINE bool operator()(int64_t e) const {
    int64_t cy = e / wc, x = e % wc;
    return raster.At(x - 1, cy) != raster.At(x, cy);
  }
};

// The vertices of corner rows y0 .. y0 + n_rows - 1, and the vertical unit edges of cell rows
// y0 .. y0 + n_edge_rows - 1.  `raster` must hold cell rows y0 - 1 .. y0 + n_rows - 1 (those that exist).
//
// Horizontal directions (1, 3): positions (y * wc + x) in increasing order in pos[d], flags in flag[d];
// strips come in order, so appending them keeps the lists sorted.
// Vertical directions (0, 2): the strips cut the columns, so the caller has to sort.  They are returned
// packed in pos[d] as (x * hc + y) * 4 + flag, not sorted; flag[d] stays empty.
// edges == nullptr: the vertical unit edges are not wanted.
template <typename T, typename Turn = TurnByValue>
void FindVerticesInStrip(const DRasterView<T>& raster, int64_t y0, int64_t n_rows, int64_t n_edge_rows,
                         int64_t wc, int64_t hc, DArray<int64_t> pos[4], DArray<uint8_t> flag[4],
                         DArray<int64_t>* edges) {
  const int64_t n = n_rows * wc;
  DArray<uint8_t> code(n);
  {
    auto out = code.DeviceView();
    ForEach(n, [=] __device__(size_t i) {
      out[i] = CornerCode<T, Turn>(raster, static_cast<int64_t>(i) % wc, y0 + static_cast<int64_t>(i) / wc);
    });
  }
  const int64_t first = y0 * wc;
  for (int d = 0; d < 4; d++) {
    int64_t count = thrust::count_if(Exec(), code.begin(), code.end(), IsVertexOf{d});
    pos[d].Reset(count);
    thrust::copy_if(Exec(), thrust::make_counting_iterator<int64_t>(first),
                    thrust::make_counting_iterator<int64_t>(first + n), code.begin(), pos[d].begin(),
                    IsVertexOf{d});
    auto p = pos[d].DeviceView();
    auto c = code.DeviceView();
    if (d == 1 || d == 3) {
      flag[d].Reset(count);
      auto f = flag[d].DeviceView();
      ForEach(count, [=] __device__(size_t i) {
        uint8_t state = (c[p[i] - first] >> (2 * d)) & 3;
        f[i] = state == 1 ? kFlagRight : (state == 3 ? kFlagDiagonal : 0);
      });
    } else {
      ForEach(count, [=] __device__(size_t i) {
        uint8_t state = (c[p[i] - first] >> (2 * d)) & 3;
        int64_t f = state == 1 ? kFlagRight : (state == 3 ? kFlagDiagonal : 0);
        int64_t x = p[i] % wc, y = p[i] / wc;
        p[i] = (x * hc + y) * 4 + f;
      });
    }
  }
  code.Free();
  if (edges == nullptr) return;
  const int64_t ne = n_edge_rows * wc;
  IsVerticalEdge<T> is_edge{raster, wc};
  int64_t count = thrust::count_if(Exec(), thrust::make_counting_iterator<int64_t>(first),
                                   thrust::make_counting_iterator<int64_t>(first + ne), is_edge);
  edges->Reset(count);
  thrust::copy_if(Exec(), thrust::make_counting_iterator<int64_t>(first),
                  thrust::make_counting_iterator<int64_t>(first + ne), edges->begin(), is_edge);
}

// The vertical vertices come out of the strips packed and unsorted.  Sort them into column order and unpack:
// packed[i] = position * 4 + flag  ->  pos[i], flag[i].  `pos` holds the packed values on entry.
inline void SortAndUnpackVertical(DArray<int64_t>* pos, DArray<uint8_t>* flag) {
  thrust::sort(Exec(), pos->begin(), pos->end());
  flag->Reset(pos->size());
  auto p = pos->DeviceView();
  auto f = flag->DeviceView();
  ForEach(p.size(), [=] __device__(size_t i) {
    f[i] = static_cast<uint8_t>(p[i] & 3);
    p[i] >>= 2;
  });
}

}  // namespace xrasp
#endif  // XRASP_STEPS_S1_CORNERS_BASE_H
