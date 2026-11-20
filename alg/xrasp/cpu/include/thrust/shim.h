#ifndef XRASP_SHIM_THRUST_SHIM_H
#define XRASP_SHIM_THRUST_SHIM_H
// A stand-in for Thrust in the CPU build, for hosts that cannot depend on Thrust (X-Rasp inside GDAL).
//
// The steps are written against Thrust: thrust::sort(Exec(), ...), thrust::make_counting_iterator, thrust::plus.
// In the CPU build the work is done by util/cpu_primitives.h; Thrust only gave the names and the dispatch.
// This directory gives the same names with nothing behind them but that dispatch: every algorithm hands its
// arguments to the function of the same name that argument-dependent lookup finds for the execution policy
// (xrasp::CpuPolicy), which is what Thrust does.  Put this directory on the include path instead of Thrust's
// (cpu/CMakeLists.txt).
#include <cstddef>
#include <iterator>
#include <utility>

// Calls made from here see no function of these names except through argument-dependent lookup.
namespace xrasp_thrust_adl {
#define XRASP_THRUST_ADL(name)                                                                \
  template <typename Policy, typename... Args>                                                \
  auto name##_(Policy& exec, Args&&... args) -> decltype(name(exec, std::forward<Args>(args)...)) { \
    return name(exec, std::forward<Args>(args)...);                                           \
  }
XRASP_THRUST_ADL(sort)
XRASP_THRUST_ADL(stable_sort_by_key)
XRASP_THRUST_ADL(inclusive_scan)
XRASP_THRUST_ADL(exclusive_scan)
XRASP_THRUST_ADL(copy_if)
XRASP_THRUST_ADL(count_if)
XRASP_THRUST_ADL(transform_reduce)
XRASP_THRUST_ADL(fill)
XRASP_THRUST_ADL(sequence)
#undef XRASP_THRUST_ADL
}  // namespace xrasp_thrust_adl

namespace thrust {

template <typename T>
struct plus {
  T operator()(const T& a, const T& b) const { return a + b; }
};

// The numbers first, first + 1, ... as a random-access iterator.
template <typename T>
class counting_iterator {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = const T*;
  using reference = T;
  counting_iterator() = default;
  explicit counting_iterator(T value) : value_(value) {}
  T operator*() const { return value_; }
  T operator[](difference_type i) const { return static_cast<T>(value_ + static_cast<T>(i)); }
  counting_iterator& operator++() {
    ++value_;
    return *this;
  }
  counting_iterator operator+(difference_type i) const { return counting_iterator(static_cast<T>(value_ + static_cast<T>(i))); }
  difference_type operator-(const counting_iterator& other) const {
    return static_cast<difference_type>(value_) - static_cast<difference_type>(other.value_);
  }
  bool operator==(const counting_iterator& other) const { return value_ == other.value_; }
  bool operator!=(const counting_iterator& other) const { return value_ != other.value_; }

 private:
  T value_{};
};

template <typename T>
counting_iterator<T> make_counting_iterator(T value) {
  return counting_iterator<T>(value);
}

namespace system {
namespace omp {
template <typename Derived>
struct execution_policy {};
}  // namespace omp
}  // namespace system

#define XRASP_THRUST_FORWARD(name)                                         \
  template <typename Policy, typename... Args>                             \
  auto name(Policy&& exec, Args&&... args) {                               \
    auto& policy = exec;                                                   \
    return xrasp_thrust_adl::name##_(policy, std::forward<Args>(args)...); \
  }
XRASP_THRUST_FORWARD(sort)
XRASP_THRUST_FORWARD(stable_sort_by_key)
XRASP_THRUST_FORWARD(inclusive_scan)
XRASP_THRUST_FORWARD(exclusive_scan)
XRASP_THRUST_FORWARD(copy_if)
XRASP_THRUST_FORWARD(count_if)
XRASP_THRUST_FORWARD(transform_reduce)
XRASP_THRUST_FORWARD(fill)
XRASP_THRUST_FORWARD(sequence)
#undef XRASP_THRUST_FORWARD

}  // namespace thrust
#endif  // XRASP_SHIM_THRUST_SHIM_H
