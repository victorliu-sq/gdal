#ifndef XRASP_UTIL_DEV_ARRAY_H
#define XRASP_UTIL_DEV_ARRAY_H
// DArray<T>: an array on the GPU, owned on the host side.  DArrayView<T>: what a kernel gets (a pointer and
// a size, passed by value).  The memory is not initialized.
#include <cassert>
#include <cstring>
#include <utility>
#include <vector>

#include <thrust/copy.h>
#include <thrust/fill.h>

#include "xrasp/util/device_memory.h"
#include "xrasp/util/macros.h"

namespace xrasp {

template <typename T>
class DArrayView {
 public:
  DArrayView() = default;
  DEV_HOST_INLINE DArrayView(T* data, size_t size) : data_(data), size_(size) {}
  DEV_INLINE T& operator[](size_t i) const { return data_[i]; }
  DEV_HOST_INLINE T* data() const { return data_; }
  DEV_HOST_INLINE size_t size() const { return size_; }
  DEV_HOST_INLINE T* begin() const { return data_; }
  DEV_HOST_INLINE T* end() const { return data_ + size_; }

 private:
  T* data_ = nullptr;
  size_t size_ = 0;
};

template <typename T>
class DArray {
 public:
  DArray() = default;
  explicit DArray(size_t size) { Allocate(size); }
  DArray(size_t size, const T& value) {
    Allocate(size);
    Fill(value);
  }
  DArray(const DArray&) = delete;
  DArray& operator=(const DArray&) = delete;
  DArray(DArray&& other) noexcept { Swap(other); }
  DArray& operator=(DArray&& other) noexcept {
    if (this != &other) {
      Free();
      Swap(other);
    }
    return *this;
  }
  ~DArray() { Free(); }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  T* data() const { return data_; }
  // raw pointers: usable as iterators in Thrust calls made with Exec()
  T* begin() const { return data_; }
  T* end() const { return data_ + size_; }
  DArrayView<T> DeviceView() const { return DArrayView<T>(data_, size_); }

  void Swap(DArray& other) {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
  }
  // Give the memory back.
  void Free() {
    if (data_ != nullptr) {
      DeviceMemory::Deallocate(data_, size_ * sizeof(T));
    }
    data_ = nullptr;
    size_ = 0;
  }
  // A new uninitialized array of `size` elements; the old contents are gone.
  void Reset(size_t size) {
    Free();
    Allocate(size);
  }
  // Keep the first `size` elements (size <= size()) in an array of exactly that size.
  void Shrink(size_t size) {
    assert(size <= size_);
    if (size == size_) return;
    DArray<T> smaller(size);
    if (size > 0) {
      CopyOnDevice(smaller.data_, data_, size * sizeof(T));
    }
    Swap(smaller);
  }
  void Fill(const T& value) { thrust::fill(Exec(), begin(), end(), value); }

  void FromHost(const T* host, size_t size) {
    Reset(size);
    if (size > 0) {
      CopyToDevice(data_, host, size * sizeof(T));
    }
  }
  void FromHost(const std::vector<T>& host) { FromHost(host.data(), host.size()); }
  void ToHost(std::vector<T>* host) const {
    host->resize(size_);
    if (size_ > 0) {
      CopyToHost(host->data(), data_, size_ * sizeof(T));
    }
  }
  std::vector<T> ToHost() const {
    std::vector<T> host;
    ToHost(&host);
    return host;
  }
  // One element, read on the host (slow: for the occasional last element of a prefix sum).
  T Get(size_t i) const {
    T value;
    CopyToHost(&value, data_ + i, sizeof(T));
    return value;
  }

 private:
  void Allocate(size_t size) {
    size_ = size;
    data_ = size == 0 ? nullptr : static_cast<T*>(DeviceMemory::Allocate(size * sizeof(T)));
  }

  T* data_ = nullptr;
  size_t size_ = 0;
};

// The cells of a strip of the raster, where the kernels read them.  On the GPU they are uploaded.  On the CPU
// the kernels read the raster where it is: nothing is copied.
template <typename T>
class StripCells {
 public:
  void Load(const T* host, size_t size) {
#ifdef XRASP_CPU
    (void)size;
    data_ = host;
#else
    array_.FromHost(host, size);
    data_ = array_.data();
#endif
  }
  const T* data() const { return data_; }

 private:
#ifndef XRASP_CPU
  DArray<T> array_;
#endif
  const T* data_ = nullptr;
};

}  // namespace xrasp
#endif  // XRASP_UTIL_DEV_ARRAY_H
