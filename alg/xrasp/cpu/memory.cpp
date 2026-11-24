// Device memory of the CPU backend: host memory, a cache line apart, with the two byte counters.
#include "xrasp/util/device_memory.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <new>
#include <optional>
#include <stdexcept>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace xrasp {

namespace {
std::atomic<int64_t> g_current{0}, g_peak{0};
uint64_t g_limit = 0;
}  // namespace

void DeviceMemory::Init(const DeviceMemoryConfig& config) { g_limit = config.limit_bytes; }

void* DeviceMemory::Allocate(size_t bytes) {
  if (g_limit > 0 && static_cast<uint64_t>(g_current.load()) + bytes > g_limit) {
    throw std::runtime_error("memory limit exceeded");
  }
  // a cache line; not zeroed; throws std::bad_alloc
  void* ptr = ::operator new((bytes + 63) / 64 * 64, std::align_val_t{64});
  int64_t now = g_current.fetch_add(static_cast<int64_t>(bytes)) + static_cast<int64_t>(bytes);
  int64_t peak = g_peak.load();
  while (now > peak && !g_peak.compare_exchange_weak(peak, now)) {
  }
  return ptr;
}

void DeviceMemory::Deallocate(void* ptr, size_t bytes) {
  ::operator delete(ptr, std::align_val_t{64});
  g_current.fetch_sub(static_cast<int64_t>(bytes));
}

int64_t DeviceMemory::CurrentBytes() { return g_current.load(); }

int64_t DeviceMemory::PeakBytes() { return g_peak.load(); }

void DeviceMemory::ResetPeak() { g_peak.store(g_current.load()); }

// glibc keeps what is freed inside its heap (an array of a band, a few megabytes, is not a mapping of its own):
// the pages of the free chunks are handed back here.
void DeviceMemory::ReturnFreed() {
#if defined(__GLIBC__)
  malloc_trim(0);
#endif
}

void DeviceMemory::ReleaseAll() {}

// DeviceBytes is in host_memory.cpp, which asks the operating system: what is here is standard C++ only and
// goes into GDAL as it is.

}  // namespace xrasp
