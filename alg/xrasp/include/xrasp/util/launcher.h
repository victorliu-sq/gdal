#ifndef XRASP_UTIL_LAUNCHER_H
#define XRASP_UTIL_LAUNCHER_H
// Kernels are written as device lambdas:
//
//   ForEach(n, [=] __device__(size_t i) mutable { out[i] = f(in[i]); });
//
// runs the lambda for i = 0 .. n-1 (views are captured by value): on the GPU one GPU thread per index, on
// the CPU the indices split evenly over the threads (util/cpu_threads.h), each thread a run of consecutive
// ones.
#include <cstddef>

#include "xrasp/util/macros.h"
#ifdef XRASP_CPU
#include "xrasp/util/cpu_threads.h"
#endif

namespace xrasp {

#ifdef XRASP_CPU

// With OpenMP the pool is OpenMP's: its threads are made once and stay (run with OMP_NUM_THREADS=n OMP_PLACES=cores
// OMP_PROC_BIND=spread, a thread per core).  Two things learnt in GACGE (src/gap/gapbs/src/bfs.cc): the team
// keeps one size, because libgomp rebuilds a bound pool when the size changes; and a small job runs on the
// calling thread (kSerialBelow, util/macros.h).

template <typename F>
void ForEach(size_t size, F f) {
  if (size < kSerialBelow) {
    for (size_t i = 0; i < size; i++) f(i);
    return;
  }
  const int parts = CpuThreads();
  RunParts(parts, [&f, size, parts](int p) {
    F mine = f;      // a copy for every thread: the lambdas are mutable
    const size_t hi = size * static_cast<size_t>(p + 1) / parts;
    for (size_t i = size * static_cast<size_t>(p) / parts; i < hi; i++) mine(i);
  });
}

#else

template <typename F>
__global__ void KernelForEach(size_t size, F f) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    f(i);
  }
}

template <typename F>
void ForEach(size_t size, F f) {
  if (size == 0) return;
  constexpr unsigned int kBlockSize = 256;
  // a grid holds at most 2^31 - 1 blocks
  size_t blocks = (size + kBlockSize - 1) / kBlockSize;
  KernelForEach<<<static_cast<unsigned int>(blocks), kBlockSize>>>(size, f);
  CUDA_CHECK(cudaGetLastError());
}

#endif

}  // namespace xrasp
#endif  // XRASP_UTIL_LAUNCHER_H
