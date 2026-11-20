#ifndef XRASP_UTIL_ALGO_H
#define XRASP_UTIL_ALGO_H
// Binary searches that run inside a kernel, on a sorted array.
#include <cstdint>

#include "xrasp/backend/macros.h"

namespace xrasp {

// The first index whose element is >= key (n when there is none).
template <typename T>
DEV_HOST_INLINE int64_t LowerBound(const T* a, int64_t n, T key) {
  int64_t lo = 0, hi = n;
  while (lo < hi) {
    int64_t mid = (lo + hi) >> 1;
    if (a[mid] < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// The first index whose element is > key (n when there is none).
template <typename T>
DEV_HOST_INLINE int64_t UpperBound(const T* a, int64_t n, T key) {
  int64_t lo = 0, hi = n;
  while (lo < hi) {
    int64_t mid = (lo + hi) >> 1;
    if (a[mid] <= key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

}  // namespace xrasp
#endif  // XRASP_UTIL_ALGO_H
