#ifndef XRASP_UTIL_DEVICE_MEMORY_H
#define XRASP_UTIL_DEVICE_MEMORY_H
// All memory the arrays of the library live in comes through this file ("device" memory: where the kernels
// run).  On the GPU: with XRASP_USE_RMM it is an RMM pool with a statistics layer on top (and, when asked, a
// limit below it), installed as RMM's current device resource so that DArray and the temporaries of Thrust
// algorithms take from the same pool; without RMM it is cudaMalloc, and the two byte counters are kept here.
// On the CPU it is host memory, not zeroed: a std::vector would have one thread write zeros
// through every new array, whereas here the threads that fill an array are the first to touch it (GAP's
// pvector, as in GACGE).  Nothing is uploaded or downloaded there, and there is nothing to wait for.
#include <cstddef>
#include <cstdint>

#include "xrasp/backend/macros.h"

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
  // Memory that was given back with Deallocate() and that the allocator still holds goes back to the
  // operating system, where the backend can do that (the CPU with glibc); nothing otherwise.
  static void ReturnFreed();
  // The GPU with a pool: when no array is live, the pool goes back to the GPU as a whole, and the next
  // Allocate() makes a new one with the configuration of the last Init().  For a caller that shares the GPU
  // with a library that has a pool of its own: between two calls X-Rasp holds no device memory.  Nothing
  // when an array is live, or without a pool.
  static void ReleaseAll();
  // Free and total memory of the GPU as the driver reports them (of the host in the CPU build).
  static void DeviceBytes(size_t* free_bytes, size_t* total_bytes);
};

}  // namespace xrasp
// Exec(), DeviceSync() and the copies between host and device memory, as the backend has them.
#include "xrasp/backend/memory.h"

#endif  // XRASP_UTIL_DEVICE_MEMORY_H
