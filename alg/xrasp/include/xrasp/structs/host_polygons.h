#ifndef XRASP_STRUCTS_HOST_POLYGONS_H
#define XRASP_STRUCTS_HOST_POLYGONS_H
// The output in host memory, for results whose vertices do not fit the GPU (mode=bands).  Rings are stored in
// the order they were finished (band after band, then the rings that cross bands), and a polygon lists its
// rings:
//
//   polygon p has value values[p] and owns rings ring_index[poly_offsets[p] .. poly_offsets[p+1]-1], the outer
//   ring first; ring r owns vertices ring_offsets[r] .. ring_offsets[r+1]-1; vertex i is corner (x[i], y[i]).
//
// With ring_index empty the rings are stored polygon by polygon (the layout of structs/polygons.h).
//
// Signature() condenses a result into 64 bits that do not depend on the order of storage: one record per ring
// (smallest corner position, length, sum of x, sum of y, outer ring or hole, the smallest corner position of
// its polygon's outer ring, the polygon's value), hashed, summed.  Two results with the same signature hold
// the same rings grouped into the same polygons.  The Python reference computes the same number.
#include <cstdint>
#include <vector>

#include "xrasp/util/host_array.h"

namespace xrasp {

struct HostPolygons {
  std::vector<int64_t> values, poly_offsets, ring_index, ring_offsets;
  HostArray<int32_t> x, y;                  // not a std::vector: filled by a copy, never initialized

  int64_t num_polygons() const { return static_cast<int64_t>(values.size()); }
  int64_t num_rings() const { return ring_offsets.empty() ? 0 : static_cast<int64_t>(ring_offsets.size()) - 1; }
  int64_t num_vertices() const { return static_cast<int64_t>(x.size()); }
};

inline uint64_t Signature(const HostPolygons& p, int64_t hc) {
  const int64_t nr = p.num_rings(), np = p.num_polygons();
  // smallest corner position of every ring (in storage order)
  std::vector<int64_t> k0(nr), sx(nr), sy(nr);
  for (int64_t r = 0; r < nr; r++) {
    int64_t kmin = INT64_MAX, ax = 0, ay = 0;
    for (int64_t i = p.ring_offsets[r]; i < p.ring_offsets[r + 1]; i++) {
      int64_t k = static_cast<int64_t>(p.x[i]) * hc + p.y[i];
      if (k < kmin) kmin = k;
      ax += p.x[i];
      ay += p.y[i];
    }
    k0[r] = kmin;
    sx[r] = ax;
    sy[r] = ay;
  }
  const bool listed = !p.ring_index.empty();
  uint64_t sum = 0;
  for (int64_t q = 0; q < np; q++) {
    const int64_t first = p.poly_offsets[q], last = p.poly_offsets[q + 1];
    const int64_t outer = listed ? p.ring_index[first] : first;
    for (int64_t s = first; s < last; s++) {
      const int64_t r = listed ? p.ring_index[s] : s;
      const int64_t fields[7] = {k0[r], p.ring_offsets[r + 1] - p.ring_offsets[r], sx[r], sy[r],
                                 s == first ? 1 : 0, k0[outer], p.values[q]};
      uint64_t h = 0x9E3779B97F4A7C15ull;
      for (int64_t f : fields) h = (h ^ static_cast<uint64_t>(f)) * 0xC2B2AE3D27D4EB4Full;
      h ^= h >> 32;
      sum += h;
    }
  }
  return sum;
}

}  // namespace xrasp
#endif  // XRASP_STRUCTS_HOST_POLYGONS_H
