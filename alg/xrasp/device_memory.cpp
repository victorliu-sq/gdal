#include "xrasp/util/device_memory.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <new>
#include <optional>
#include <stdexcept>

#ifdef XRASP_USE_RMM
#include <rmm/aligned.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#endif

namespace xrasp {

#if defined(XRASP_CPU)

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

// DeviceBytes of the CPU build is in run_polygonize.cu, its one caller: what is here is standard C++ only and
// goes into GDAL as it is.

#elif defined(XRASP_USE_RMM)

namespace {
// The statistics layer is on top: it counts the bytes in live arrays (what the algorithm needs), not what the
// pool holds.  Copies of an RMM resource share their state, so this handle reads the installed resource.
std::optional<rmm::mr::statistics_resource_adaptor> g_stats;
}  // namespace

void DeviceMemory::Init(const DeviceMemoryConfig& config) {
  using any_resource = cuda::mr::any_resource<cuda::mr::device_accessible>;
  any_resource upstream{rmm::mr::cuda_memory_resource{}};
  if (config.pool) {
    // start small and grow on demand; with a limit the pool never takes more than that from the GPU
    const std::size_t align = 256;
    std::size_t initial = std::size_t{256} << 20;
    std::optional<std::size_t> maximum;
    if (config.limit_bytes > 0) {
      maximum = static_cast<std::size_t>(config.limit_bytes) / align * align;
      initial = std::min(initial, *maximum);
    }
    upstream = any_resource{rmm::mr::pool_memory_resource{std::move(upstream), initial, maximum}};
  } else if (config.limit_bytes > 0) {
    upstream = any_resource{
        rmm::mr::limiting_resource_adaptor{std::move(upstream), static_cast<std::size_t>(config.limit_bytes)}};
  }
  g_stats.emplace(std::move(upstream));
  rmm::mr::set_current_device_resource(any_resource{*g_stats});
}

void* DeviceMemory::Allocate(size_t bytes) {
  if (!g_stats) Init(DeviceMemoryConfig{});
  return g_stats->allocate_sync(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

void DeviceMemory::Deallocate(void* ptr, size_t bytes) {
  g_stats->deallocate_sync(ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

int64_t DeviceMemory::CurrentBytes() { return g_stats ? g_stats->get_bytes_counter().value : 0; }

int64_t DeviceMemory::PeakBytes() { return g_stats ? g_stats->get_bytes_counter().peak : 0; }

void DeviceMemory::ResetPeak() {
  if (!g_stats) return;
  // a fresh pair of counters on top of the stack: its peak starts from what is live now
  auto before = g_stats->push_counters();
  (void)before;
}

#else  // plain cudaMalloc

namespace {
std::atomic<int64_t> g_current{0}, g_peak{0};
uint64_t g_limit = 0;
}  // namespace

void DeviceMemory::Init(const DeviceMemoryConfig& config) { g_limit = config.limit_bytes; }

void* DeviceMemory::Allocate(size_t bytes) {
  if (g_limit > 0 && static_cast<uint64_t>(g_current.load()) + bytes > g_limit) {
    throw std::runtime_error("GPU memory limit exceeded");
  }
  void* ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, bytes));
  int64_t now = g_current.fetch_add(static_cast<int64_t>(bytes)) + static_cast<int64_t>(bytes);
  int64_t peak = g_peak.load();
  while (now > peak && !g_peak.compare_exchange_weak(peak, now)) {
  }
  return ptr;
}

void DeviceMemory::Deallocate(void* ptr, size_t bytes) {
  CUDA_CHECK(cudaFree(ptr));
  g_current.fetch_sub(static_cast<int64_t>(bytes));
}

int64_t DeviceMemory::CurrentBytes() { return g_current.load(); }

int64_t DeviceMemory::PeakBytes() { return g_peak.load(); }

void DeviceMemory::ResetPeak() { g_peak.store(g_current.load()); }

#endif

#ifndef XRASP_CPU
void DeviceMemory::DeviceBytes(size_t* free_bytes, size_t* total_bytes) {
  CUDA_CHECK(cudaMemGetInfo(free_bytes, total_bytes));
}
#endif

}  // namespace xrasp
