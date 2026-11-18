#ifndef XRASP_STRUCTS_RINGS_H
#define XRASP_STRUCTS_RINGS_H
// What is known about the rings once every vertex has its ring and its position in it.
#include <cstdint>

#include "xrasp/util/dev_array.h"

namespace xrasp {

struct Rings {
  int64_t nv = 0, nr = 0, nholes = 0;
  // per vertex
  DArray<int32_t> ring;       // ring number: rings are numbered by their smallest vertex
  DArray<int32_t> rank;       // position in the ring, 0 at the ring's smallest vertex
  // per ring
  DArray<int32_t> length;     // number of vertices
  DArray<uint8_t> is_hole;    // 1: the ring is a hole (the block is outside it)
  DArray<int32_t> cell_x;     // a cell of the ring's own block (its value is the ring's value)
  DArray<int32_t> cell_y;
  DArray<int32_t> parent;     // the outer ring of the ring's polygon (itself for an outer ring)
  // per hole
  DArray<int32_t> holes;      // ring numbers of the holes
  DArray<int64_t> hole_cx;    // the cell just west of the top end of the hole's leftmost edge:
  DArray<int64_t> hole_cy;    // a cell of the hole's block
};

}  // namespace xrasp
#endif  // XRASP_STRUCTS_RINGS_H
