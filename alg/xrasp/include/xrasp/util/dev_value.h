#ifndef XRASP_UTIL_DEV_VALUE_H
#define XRASP_UTIL_DEV_VALUE_H
// DValue<T>: one value on the GPU (a counter, a "something changed" flag), with its view for kernels.
#include "xrasp/util/dev_array.h"

namespace xrasp {

template <typename T>
class DValueView {
 public:
  DEV_HOST_INLINE explicit DValueView(T* data) : data_(data) {}
#ifdef XRASP_CPU
  // Many threads set the same flag.  A write takes the cache line away from every other core, a read does
  // not: write only when the value is not there yet.
  DEV_INLINE void Set(const T& value) const {
    if (*data_ != value) *data_ = value;
  }
#else
  DEV_INLINE void Set(const T& value) const { *data_ = value; }
#endif
  DEV_INLINE T Value() const { return *data_; }
  DEV_HOST_INLINE T* data() const { return data_; }

 private:
  T* data_;
};

template <typename T>
class DValue {
 public:
  DValue() : array_(1) {}
  explicit DValue(const T& value) : array_(1) { SetH2D(value); }
  void SetH2D(const T& value) { CopyToDevice(array_.data(), &value, sizeof(T)); }
  T GetD2H() const { return array_.Get(0); }
  DValueView<T> DeviceView() const { return DValueView<T>(array_.data()); }

 private:
  DArray<T> array_;
};

}  // namespace xrasp
#endif  // XRASP_UTIL_DEV_VALUE_H
