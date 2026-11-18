#ifndef XRASP_UTIL_CPU_THREADS_H
#define XRASP_UTIL_CPU_THREADS_H
// The threads of the CPU build (XRASP_CPU).  Everything parallel there comes down to one call,
//
//   RunParts(parts, [&](int part) { ... });
//
// which runs the body for part = 0 .. parts-1, each on a thread of its own, and returns when all are done
// (util/launcher.h and util/cpu_primitives.h cut their ranges into parts on top of it).  The threads are
//
//   OpenMP's          when the compiler was given OpenMP;
//   the host's pool   otherwise, when the host has named one with SetCpuPool() -- inside GDAL the worker
//                     threads GDAL already keeps (alg/polygonize_xrasp.cpp), so that X-Rasp needs nothing
//                     GDAL does not have;
//   new threads       otherwise, made for the call and joined (the tests without OpenMP).
//
// CpuThreads() is the number of parts a job is cut into, SetCpuThreads() sets it for the calling thread (as
// omp_set_num_threads does).  A RunParts inside a part runs its parts one after the other on that thread.
// SetCpuPool() is there in both builds, so that the host names its pool without asking which one it is in;
// with OpenMP the pool is not used.
#ifdef _OPENMP
#include <omp.h>
#else
#include <atomic>
#include <exception>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>
#endif

namespace xrasp {

// One part of a job: fn(job, part).
typedef void (*CpuPartFn)(void* job, int part);
// A pool: calls fn(job, part) for part = 0 .. parts-1, each once, on threads of its choice (the calling
// thread may take some), and returns when all the calls have returned.
typedef void (*CpuPoolFn)(int parts, CpuPartFn fn, void* job);

#ifdef _OPENMP

inline void SetCpuPool(CpuPoolFn) {}
inline const char* CpuThreadsOf() { return "OpenMP"; }
inline int CpuThreads() { return omp_get_max_threads(); }
inline void SetCpuThreads(int threads) { omp_set_num_threads(threads); }

template <typename Body>
void RunParts(int parts, Body body) {
  if (parts <= 1) {
    if (parts == 1) body(0);
    return;
  }
#pragma omp parallel for schedule(static, 1) num_threads(parts)
  for (int p = 0; p < parts; p++) body(p);
}

#else

namespace cpu_threads {
inline std::atomic<CpuPoolFn>& Pool() {
  static std::atomic<CpuPoolFn> pool{nullptr};
  return pool;
}
inline int& Threads() {
  static thread_local int threads = 0;      // 0: not set on this thread
  return threads;
}
inline bool& InPart() {
  static thread_local bool in_part = false;
  return in_part;
}
// Without a pool: a new thread for every part but the first.
inline void RunOnNewThreads(int parts, CpuPartFn fn, void* job) {
  std::vector<std::thread> threads;
  threads.reserve(parts - 1);
  for (int p = 1; p < parts; p++) {
    try {
      threads.emplace_back(fn, job, p);
    } catch (const std::system_error&) {
      fn(job, p);      // no thread to be had
    }
  }
  fn(job, 0);
  for (std::thread& t : threads) t.join();
}
}  // namespace cpu_threads

inline const char* CpuThreadsOf() { return cpu_threads::Pool().load() != nullptr ? "the host's pool" : "new threads"; }
inline void SetCpuPool(CpuPoolFn pool) { cpu_threads::Pool().store(pool); }
inline int CpuThreads() {
  if (cpu_threads::Threads() > 0) return cpu_threads::Threads();
  const unsigned int n = std::thread::hardware_concurrency();
  return n > 0 ? static_cast<int>(n) : 1;
}
inline void SetCpuThreads(int threads) { cpu_threads::Threads() = threads; }

template <typename Body>
void RunParts(int parts, Body body) {
  if (parts <= 1 || cpu_threads::InPart()) {
    for (int p = 0; p < parts; p++) body(p);
    return;
  }
  // what a part throws is thrown again by the calling thread (the first one that was caught)
  struct Job {
    Body* body;
    std::mutex mutex;
    std::exception_ptr error;
  } job{&body, {}, nullptr};
  const CpuPartFn fn = [](void* j, int part) {
    Job* job = static_cast<Job*>(j);
    const bool before = cpu_threads::InPart();
    cpu_threads::InPart() = true;
    try {
      (*job->body)(part);
    } catch (...) {
      std::lock_guard<std::mutex> lock(job->mutex);
      if (!job->error) job->error = std::current_exception();
    }
    cpu_threads::InPart() = before;
  };
  const CpuPoolFn pool = cpu_threads::Pool().load();
  (pool != nullptr ? pool : &cpu_threads::RunOnNewThreads)(parts, fn, &job);
  if (job.error) std::rethrow_exception(job.error);
}

#endif

}  // namespace xrasp
#endif  // XRASP_UTIL_CPU_THREADS_H
