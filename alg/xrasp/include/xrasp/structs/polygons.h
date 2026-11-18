#ifndef XRASP_STRUCTS_POLYGONS_H
#define XRASP_STRUCTS_POLYGONS_H
// The output, in the layout of the reference implementation (and of GeoArrow polygons):
//
//   polygon p has value values[p] and owns rings poly_offsets[p] .. poly_offsets[p+1]-1, the outer ring first;
//   ring r owns vertices ring_offsets[r] .. ring_offsets[r+1]-1 (not closed); vertex i is corner (x[i], y[i]).
//
// Checksum() condenses one array into 64 bits that depend on every element and its place; the reference
// implementation computes the same number (measure/.../reference_result.py), so equal checksums of all five
// arrays mean the outputs are identical element by element.
#include <cstdint>
#include <cstdio>
#include <string>

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include "xrasp/util/dev_array.h"

namespace xrasp {

template <typename T>
struct ChecksumTerm {
  const T* a;
  DEV_HOST_INLINE uint64_t operator()(int64_t i) const {
    uint64_t v = static_cast<uint64_t>(static_cast<int64_t>(a[i]));
    return (v ^ (static_cast<uint64_t>(i + 1) * 0x9E3779B97F4A7C15ull)) * 0xC2B2AE3D27D4EB4Full;
  }
};

template <typename T>
inline uint64_t Checksum(const DArray<T>& a) {
  return thrust::transform_reduce(Exec(), thrust::make_counting_iterator<int64_t>(0),
                                  thrust::make_counting_iterator<int64_t>(static_cast<int64_t>(a.size())),
                                  ChecksumTerm<T>{a.data()}, static_cast<uint64_t>(0), thrust::plus<uint64_t>());
}

inline std::string Hex(uint64_t v) {
  char buf[19];
  snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v));
  return std::string(buf);
}

struct Polygons {
  DArray<int64_t> values;
  DArray<int64_t> poly_offsets;
  DArray<int64_t> ring_offsets;
  DArray<int32_t> x;
  DArray<int32_t> y;

  int64_t num_polygons() const { return static_cast<int64_t>(values.size()); }
  int64_t num_rings() const { return ring_offsets.empty() ? 0 : static_cast<int64_t>(ring_offsets.size()) - 1; }
  int64_t num_vertices() const { return static_cast<int64_t>(x.size()); }
};

}  // namespace xrasp
#endif  // XRASP_STRUCTS_POLYGONS_H
