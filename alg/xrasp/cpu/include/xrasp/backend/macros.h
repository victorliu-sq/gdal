#ifndef XRASP_BACKEND_MACROS_H
#define XRASP_BACKEND_MACROS_H
// The CPU backend.  The library has two backends, cpu/ and gpu/, with headers of the same names under
// xrasp/backend/: a build puts one of the two directories on its include path, and that is the whole choice
// -- the steps (include/xrasp/steps) are the same text for both and name no backend.
//
// Here a kernel is the same lambda in a loop over the indices, split over the threads (launcher.h,
// threads.h), and "device" memory is host memory.  Standard C++ only: no CUDA, no Thrust (the names of
// Thrust the steps use are in cpu/include/thrust and do their work in primitives.h).
#include <cstdio>
#include <cstdlib>

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

namespace xrasp {
// A flag that many threads set to the same value.  A write takes the cache line away from every other core,
// a read does not: write only when the value is not there yet.
template <typename T>
inline void SetShared(T* address, const T& value) {
  if (*address != value) *address = value;
}

// The kernels read host memory: the cells of a raster are used where they are, nothing is uploaded.
constexpr bool kKernelsReadHostMemory = true;
}  // namespace xrasp

#endif  // XRASP_BACKEND_MACROS_H
