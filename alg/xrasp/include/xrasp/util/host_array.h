#ifndef XRASP_UTIL_HOST_ARRAY_H
#define XRASP_UTIL_HOST_ARRAY_H
// An array in host memory whose elements are not initialized when it is made: for the large arrays that are
// filled right away by a copy (the vertices of a result).  A std::vector writes zeros first, on one thread.
#include <cstddef>
#include <cstdlib>
#include <new>
#include <utility>

namespace xrasp {

template <typename T>
class HostArray {
 public:
  HostArray() = default;
  explicit HostArray(size_t size) { Reset(size); }
  HostArray(const HostArray&) = delete;
  HostArray& operator=(const HostArray&) = delete;
  HostArray(HostArray&& other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }
  HostArray& operator=(HostArray&& other) noexcept {
    if (this != &other) {
      std::free(data_);
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }
  ~HostArray() { std::free(data_); }

  // size elements, not initialized; what was there is gone
  void Reset(size_t size) {
    std::free(data_);
    data_ = nullptr;
    size_ = 0;
    if (size > 0) {
      data_ = static_cast<T*>(std::malloc(size * sizeof(T)));
      if (data_ == nullptr) throw std::bad_alloc();
      size_ = size;
    }
  }
  void Free() { Reset(0); }

  T* data() { return data_; }
  const T* data() const { return data_; }
  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  T& operator[](size_t i) { return data_[i]; }
  const T& operator[](size_t i) const { return data_[i]; }
  T* begin() { return data_; }
  T* end() { return data_ + size_; }
  const T* begin() const { return data_; }
  const T* end() const { return data_ + size_; }

 private:
  T* data_ = nullptr;
  size_t size_ = 0;
};

}  // namespace xrasp
#endif  // XRASP_UTIL_HOST_ARRAY_H
