#ifndef XRASP_UTIL_DEVICE_MEMORY_H
#define XRASP_UTIL_DEVICE_MEMORY_H
// All memory the arrays of the library live in comes through this file ("device" memory: where the kernels
// run).  On the GPU: with XRASP_USE_RMM it is an RMM pool with a statistics layer on top (and, when asked, a
// limit below it), installed as RMM's current device resource so that DArray and the temporaries of Thrust
// algorithms take from the same pool; without RMM it is cudaMalloc, and the two byte counters are kept here.
// On the CPU (XRASP_CPU) it is host memory, not zeroed: a std::vector would have one thread write zeros
// through every new array, whereas here the threads that fill an array are the first to touch it (GAP's
// pvector, as in GACGE).  Nothing is uploaded or downloaded there, and there is nothing to wait for.
#include <cstddef>
#include <cstdint>

#include "xrasp/util/macros.h"
#ifdef XRASP_CPU
#include "xrasp/util/cpu_primitives.h"
#else
#include <thrust/execution_policy.h>
#ifdef XRASP_USE_RMM
#include <rmm/exec_policy.hpp>
#endif
#endif

namespace xrasp {

struct DeviceMemoryConfig {
  bool pool = true;             // take memory from a pool instead of one cudaMalloc per array
  uint64_t limit_bytes = 0;     // 0: no limit.  Otherwise an allocation that would exceed it throws
};

class DeviceMemory {
 public:
  // Install the memory resources.  Call once, before the first DArray.
  static void Init(const DeviceMemoryConfig& config);
  static void* Allocate(size_t bytes);
  static void Deallocate(void* ptr, size_t bytes);
  // Bytes in live arrays now, and the most there were since the last ResetPeak().
  static int64_t CurrentBytes();
  static int64_t PeakBytes();
  static void ResetPeak();
  // Free and total memory of the GPU as the driver reports them (of the host in the CPU build).
  static void DeviceBytes(size_t* free_bytes, size_t* total_bytes);
};

// The execution policy every Thrust call of the library uses: on the GPU, temporaries from the pool; on the
// CPU, the functions of util/cpu_primitives.h.
#if defined(XRASP_CPU)
inline CpuPolicy Exec() { return CpuPolicy(); }
#elif defined(XRASP_USE_RMM)
inline auto Exec() { return rmm::exec_policy(); }
#else
inline auto Exec() { return thrust::device; }
#endif

// Wait until the kernels launched so far are done (a kernel on the CPU returns when it is done).
#ifdef XRASP_CPU
inline void DeviceSync() {}
#else
inline void DeviceSync() { CUDA_CHECK(cudaDeviceSynchronize()); }
#endif

// Copies between host memory and device memory, and within device memory.
#ifdef XRASP_CPU
inline void CopyToDevice(void* device, const void* host, size_t bytes) { ParallelCopy(device, host, bytes); }
inline void CopyToHost(void* host, const void* device, size_t bytes) { ParallelCopy(host, device, bytes); }
inline void CopyOnDevice(void* to, const void* from, size_t bytes) { ParallelCopy(to, from, bytes); }
#else
inline void CopyToDevice(void* device, const void* host, size_t bytes) {
  CUDA_CHECK(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
}
inline void CopyToHost(void* host, const void* device, size_t bytes) {
  CUDA_CHECK(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost));
}
inline void CopyOnDevice(void* to, const void* from, size_t bytes) {
  CUDA_CHECK(cudaMemcpy(to, from, bytes, cudaMemcpyDeviceToDevice));
}
#endif

}  // namespace xrasp
#endif  // XRASP_UTIL_DEVICE_MEMORY_H
