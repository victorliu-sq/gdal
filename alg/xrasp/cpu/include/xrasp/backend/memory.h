#ifndef XRASP_BACKEND_MEMORY_H
#define XRASP_BACKEND_MEMORY_H
// "Device" memory on the CPU is host memory (src/memory.cpp): nothing is uploaded or downloaded, and there
// is nothing to wait for.
#include <cstddef>

#include "xrasp/backend/primitives.h"
#include "xrasp/backend/threads.h"

namespace xrasp {

// The execution policy every Thrust call of the library uses: the functions of primitives.h.
inline CpuPolicy Exec() { return CpuPolicy(); }

// Wait until the kernels launched so far are done: a kernel on the CPU returns when it is done.
inline void DeviceSync() {}

// Copies between host memory and device memory, and within device memory.
inline void CopyToDevice(void* device, const void* host, size_t bytes) { ParallelCopy(device, host, bytes); }
inline void CopyToHost(void* host, const void* device, size_t bytes) { ParallelCopy(host, device, bytes); }
inline void CopyOnDevice(void* to, const void* from, size_t bytes) { ParallelCopy(to, from, bytes); }

// What a run reports of where it ran.
inline const char* BackendName() { return "cpu"; }
inline int BackendThreads() { return CpuThreads(); }

}  // namespace xrasp
#endif  // XRASP_BACKEND_MEMORY_H
