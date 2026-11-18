#ifndef XRASP_UTIL_MACROS_H
#define XRASP_UTIL_MACROS_H
// The library builds two ways from the same sources:
//
//   on the GPU (the default)   CUDA: a kernel is a device lambda, one GPU thread per index
//   on the CPU (XRASP_CPU)     OpenMP: the same lambda in a parallel loop over the indices
//
// The steps (steps/*) are the same text in both.  What differs is in util/: this file, launcher.h (how a
// kernel runs), device_memory.h (where the arrays live, and the sort, the prefix sum and the selection on
// the CPU) and dev_array.h.
#include <cstdio>
#include <cstdlib>

#ifdef XRASP_CPU

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// A function that runs "on the device" is an ordinary inline function.  Standard C++ only, so that every
// compiler GDAL is built with takes it.
#define DEV_INLINE inline
#define DEV_HOST
#define DEV_HOST_INLINE inline
// the kernels are written `[=] __device__(size_t i) { ... }`
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif

namespace xrasp {
// An input smaller than this is handled by the calling thread: a parallel region and its barrier cost tens
// of microseconds (GACGE, src/gap/gapbs/src/bfs.cc).
constexpr size_t kSerialBelow = 4096;
}  // namespace xrasp

// CUDA's atomics, with std::atomic: the element of an ordinary array is used as an atomic of the same size
// (lock-free, so the same instructions as the compilers' built-ins).  Each returns the value that was there
// before.  (SameAs<T>: the type
// is taken from the pointer alone, as with CUDA's overloads; std::type_identity_t without C++20.)
namespace xrasp {
template <typename T>
struct SameAsT {
  using type = T;
};
template <typename T>
using SameAs = typename SameAsT<T>::type;
}  // namespace xrasp
namespace xrasp {
template <typename T>
std::atomic<T>* AsAtomic(T* address) {
  static_assert(sizeof(std::atomic<T>) == sizeof(T) && std::atomic<T>::is_always_lock_free,
                "an atomic of the size of the element");
  return reinterpret_cast<std::atomic<T>*>(address);
}
}  // namespace xrasp
template <typename T>
inline T atomicCAS(T* address, xrasp::SameAs<T> compare, xrasp::SameAs<T> value) {
  xrasp::AsAtomic(address)->compare_exchange_strong(compare, value);
  return compare;      // the value that was there: compare itself when the exchange took place
}
template <typename T>
inline T atomicAdd(T* address, xrasp::SameAs<T> value) {
  return xrasp::AsAtomic(address)->fetch_add(value);
}
template <typename T>
inline T atomicMin(T* address, xrasp::SameAs<T> value) {
  std::atomic<T>* a = xrasp::AsAtomic(address);
  T old = a->load(std::memory_order_relaxed);
  while (value < old && !a->compare_exchange_weak(old, value)) {
  }
  return old;
}

#else  // the GPU

#include <cuda_runtime.h>

// A function that runs on the GPU / on both sides, always inlined.
#define DEV_INLINE __device__ __forceinline__
#define DEV_HOST __host__ __device__
#define DEV_HOST_INLINE __host__ __device__ __forceinline__

#define CUDA_CHECK(call)                                                                  \
  do {                                                                                    \
    cudaError_t xrasp_err_ = (call);                                                      \
    if (xrasp_err_ != cudaSuccess) {                                                      \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,          \
              cudaGetErrorString(xrasp_err_));                                            \
      abort();                                                                            \
    }                                                                                     \
  } while (0)

#endif

#endif  // XRASP_UTIL_MACROS_H
