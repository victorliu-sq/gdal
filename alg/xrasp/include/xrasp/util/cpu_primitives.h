#ifndef XRASP_UTIL_CPU_PRIMITIVES_H
#define XRASP_UTIL_CPU_PRIMITIVES_H
// The CPU build (XRASP_CPU): the sort, the prefix sum, the selection and the reductions the steps ask Thrust
// for, on the threads of util/cpu_threads.h (OpenMP's, or the host's pool).
//
// The steps call thrust::sort(Exec(), ...), thrust::inclusive_scan(Exec(), ...) and so on.  On the GPU Exec()
// is Thrust's CUDA policy.  Here it is CpuPolicy, and the functions below are found for it by
// argument-dependent lookup (Thrust's way of plugging in a backend), so the steps stay as they are and the
// algorithm keeps its primitives:
//
//   prefix sum    blocks, one per thread: the sum of every block in parallel, the running total over the
//                 blocks by one thread, then every block filled in parallel (GAP's ParallelPrefixSum, as
//                 in GACGE src/gap/gapbs/src/builder.h)
//   selection     every thread counts what it keeps of its range, a running total says where each thread
//                 writes, every thread writes; the output keeps the input order
//   sort          least-significant-digit radix sort, 8 bits a pass, a histogram per thread; stable;
//                 passes over bytes that are zero in every key are skipped (the GPU sort is a radix sort
//                 too)
//   count, sum    a partial result per thread
//
// Small inputs run on the calling thread (kSerialBelow, util/macros.h).
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <thrust/functional.h>
#include <thrust/system/omp/execution_policy.h>

#include "xrasp/util/cpu_threads.h"
#include "xrasp/util/macros.h"

namespace xrasp {

struct CpuPolicy : thrust::system::omp::execution_policy<CpuPolicy> {};

// Number of ranges an input of n elements is cut into: one per thread, or one in all when it is small.
inline int CpuParts(size_t n) { return n < kSerialBelow ? 1 : CpuThreads(); }

// body(part, lo, hi) for every part; the parts are consecutive ranges of 0 .. n-1 in order.
template <typename Body>
void ForParts(size_t n, int parts, Body body) {
  if (parts == 1) {
    body(0, static_cast<size_t>(0), n);
    return;
  }
  RunParts(parts, [&body, n, parts](int p) {
    body(p, n * static_cast<size_t>(p) / parts, n * static_cast<size_t>(p + 1) / parts);
  });
}

inline void ParallelCopy(void* to, const void* from, size_t bytes) {
  ForParts(bytes, bytes < (size_t{1} << 20) ? 1 : CpuThreads(), [=](int, size_t lo, size_t hi) {
    std::memcpy(static_cast<char*>(to) + lo, static_cast<const char*>(from) + lo, hi - lo);
  });
}

// ---- prefix sums ------------------------------------------------------------------------------------------
// result[i] = init op in[0] op ... op in[i-1] (exclusive) or ... op in[i] (inclusive, has_init says whether
// init takes part).  result may be the input.
template <typename In, typename Out, typename T, typename Op>
Out ScanBlocks(In first, In last, Out result, T init, bool has_init, bool inclusive, Op op) {
  const size_t n = static_cast<size_t>(last - first);
  if (n == 0) return result;
  const int parts = CpuParts(n);
  std::vector<T> sum(parts), before(parts);
  ForParts(n, parts, [&](int p, size_t lo, size_t hi) {
    T s = first[lo];
    for (size_t i = lo + 1; i < hi; i++) s = op(s, static_cast<T>(first[i]));
    sum[p] = s;
  });
  before[0] = init;
  for (int p = 1; p < parts; p++) {
    before[p] = (p == 1 && !has_init) ? sum[0] : op(before[p - 1], sum[p - 1]);
  }
  ForParts(n, parts, [&](int p, size_t lo, size_t hi) {
    T acc = before[p];
    size_t i = lo;
    if (p == 0 && !has_init) {                 // inclusive without init: starts from the first element
      acc = first[lo];
      result[lo] = acc;
      i = lo + 1;
    }
    for (; i < hi; i++) {
      T v = first[i];
      if (inclusive) {
        acc = op(acc, v);
        result[i] = acc;
      } else {
        result[i] = acc;
        acc = op(acc, v);
      }
    }
  });
  return result + n;
}

template <typename In, typename Out, typename Op>
Out inclusive_scan(CpuPolicy&, In first, In last, Out result, Op op) {
  using T = typename std::iterator_traits<In>::value_type;
  return ScanBlocks(first, last, result, T{}, false, true, op);
}
template <typename In, typename Out>
Out inclusive_scan(CpuPolicy& exec, In first, In last, Out result) {
  using T = typename std::iterator_traits<In>::value_type;
  return inclusive_scan(exec, first, last, result, thrust::plus<T>());
}
template <typename In, typename Out, typename T, typename Op>
Out exclusive_scan(CpuPolicy&, In first, In last, Out result, T init, Op op) {
  return ScanBlocks(first, last, result, init, true, false, op);
}
template <typename In, typename Out, typename T>
Out exclusive_scan(CpuPolicy& exec, In first, In last, Out result, T init) {
  return exclusive_scan(exec, first, last, result, init, thrust::plus<T>());
}
template <typename In, typename Out>
Out exclusive_scan(CpuPolicy& exec, In first, In last, Out result) {
  using T = typename std::iterator_traits<In>::value_type;
  return exclusive_scan(exec, first, last, result, T{}, thrust::plus<T>());
}

// ---- selection --------------------------------------------------------------------------------------------
// The elements first[i] with keep(i), in order.
template <typename In, typename Out, typename Keep>
Out SelectInOrder(In first, In last, Out result, Keep keep) {
  const size_t n = static_cast<size_t>(last - first);
  if (n == 0) return result;
  const int parts = CpuParts(n);
  std::vector<size_t> start(parts + 1, 0);
  ForParts(n, parts, [&](int p, size_t lo, size_t hi) {
    size_t k = 0;
    for (size_t i = lo; i < hi; i++) k += keep(i) ? 1 : 0;
    start[p + 1] = k;
  });
  for (int p = 0; p < parts; p++) start[p + 1] += start[p];
  ForParts(n, parts, [&](int p, size_t lo, size_t hi) {
    size_t at = start[p];
    for (size_t i = lo; i < hi; i++) {
      if (keep(i)) result[at++] = first[i];
    }
  });
  return result + start[parts];
}

template <typename In, typename Out, typename Pred>
Out copy_if(CpuPolicy&, In first, In last, Out result, Pred pred) {
  return SelectInOrder(first, last, result, [=](size_t i) mutable { return static_cast<bool>(pred(first[i])); });
}
template <typename In, typename Stencil, typename Out, typename Pred>
Out copy_if(CpuPolicy&, In first, In last, Stencil stencil, Out result, Pred pred) {
  return SelectInOrder(first, last, result, [=](size_t i) mutable { return static_cast<bool>(pred(stencil[i])); });
}

// ---- reductions and fills -----------------------------------------------------------------------------------
template <typename In, typename Fn, typename T, typename Op>
T transform_reduce(CpuPolicy&, In first, In last, Fn fn, T init, Op op) {
  const size_t n = static_cast<size_t>(last - first);
  if (n == 0) return init;
  const int parts = CpuParts(n);
  std::vector<T> part(parts);
  ForParts(n, parts, [&](int p, size_t lo, size_t hi) {
    Fn f = fn;
    T s = f(first[lo]);
    for (size_t i = lo + 1; i < hi; i++) s = op(s, static_cast<T>(f(first[i])));
    part[p] = s;
  });
  T total = init;
  for (int p = 0; p < parts; p++) total = op(total, part[p]);
  return total;
}

template <typename In, typename Pred>
typename std::iterator_traits<In>::difference_type count_if(CpuPolicy& exec, In first, In last, Pred pred) {
  using D = typename std::iterator_traits<In>::difference_type;
  return transform_reduce(
      exec, first, last, [=](const auto& v) mutable { return static_cast<D>(pred(v) ? 1 : 0); }, D{0},
      thrust::plus<D>());
}

template <typename It, typename T>
void fill(CpuPolicy&, It first, It last, const T& value) {
  const size_t n = static_cast<size_t>(last - first);
  ForParts(n, CpuParts(n), [&](int, size_t lo, size_t hi) {
    for (size_t i = lo; i < hi; i++) first[i] = value;
  });
}

template <typename It, typename T>
void sequence(CpuPolicy&, It first, It last, T init) {
  using V = typename std::iterator_traits<It>::value_type;
  const size_t n = static_cast<size_t>(last - first);
  ForParts(n, CpuParts(n), [&](int, size_t lo, size_t hi) {
    for (size_t i = lo; i < hi; i++) first[i] = static_cast<V>(static_cast<V>(init) + static_cast<V>(i));
  });
}

// ---- sort ---------------------------------------------------------------------------------------------------
// The key as an unsigned number with the same order.
template <typename K>
inline std::make_unsigned_t<K> RadixKey(K k) {
  using U = std::make_unsigned_t<K>;
  return std::is_signed_v<K> ? static_cast<U>(static_cast<U>(k) ^ (U{1} << (sizeof(K) * 8 - 1))) : static_cast<U>(k);
}

// Sorts keys[0 .. n-1] (and vals with them, unless vals is null) by key, stably.
template <typename K, typename V>
void RadixSort(K* keys, V* vals, size_t n) {
  using U = std::make_unsigned_t<K>;
  if (n < 2) return;
  if (n < kSerialBelow) {
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return keys[a] < keys[b]; });
    std::vector<K> k2(n);
    for (size_t i = 0; i < n; i++) k2[i] = keys[order[i]];
    if (vals != nullptr) {
      std::vector<V> v2(n);
      for (size_t i = 0; i < n; i++) v2[i] = vals[order[i]];
      std::copy(v2.begin(), v2.end(), vals);
    }
    std::copy(k2.begin(), k2.end(), keys);
    return;
  }
  const int parts = CpuThreads();
  // which bytes differ between keys: the others need no pass
  std::vector<U> ors(parts, 0), ands(parts, static_cast<U>(~U{0}));
  ForParts(n, parts, [&](int p, size_t lo, size_t hi) {
    U o = 0, a = static_cast<U>(~U{0});
    for (size_t i = lo; i < hi; i++) {
      U u = RadixKey(keys[i]);
      o |= u;
      a &= u;
    }
    ors[p] = o;
    ands[p] = a;
  });
  U differ = 0;
  {
    U o = 0, a = static_cast<U>(~U{0});
    for (int p = 0; p < parts; p++) {
      o |= ors[p];
      a &= ands[p];
    }
    differ = static_cast<U>(o & ~a);
  }
  std::unique_ptr<K[]> key_buf(new K[n]);                      // not zeroed: first written by the threads
  std::unique_ptr<V[]> val_buf(vals != nullptr ? new V[n] : nullptr);
  K* src_k = keys;
  K* dst_k = key_buf.get();
  V* src_v = vals;
  V* dst_v = val_buf.get();
  std::vector<size_t> hist(static_cast<size_t>(parts) * 256);
  for (size_t byte = 0; byte < sizeof(K); byte++) {
    const int shift = static_cast<int>(byte * 8);
    if (((differ >> shift) & 0xff) == 0) continue;
    ForParts(n, parts, [&](int p, size_t lo, size_t hi) {
      size_t local[256] = {0};
      for (size_t i = lo; i < hi; i++) local[(RadixKey(src_k[i]) >> shift) & 0xff]++;
      std::copy(local, local + 256, hist.begin() + static_cast<size_t>(p) * 256);
    });
    size_t at = 0;
    for (int b = 0; b < 256; b++) {
      for (int p = 0; p < parts; p++) {
        size_t c = hist[static_cast<size_t>(p) * 256 + b];
        hist[static_cast<size_t>(p) * 256 + b] = at;
        at += c;
      }
    }
    ForParts(n, parts, [&](int p, size_t lo, size_t hi) {
      size_t local[256];
      std::copy(hist.begin() + static_cast<size_t>(p) * 256, hist.begin() + static_cast<size_t>(p + 1) * 256, local);
      if (src_v != nullptr) {
        for (size_t i = lo; i < hi; i++) {
          size_t to = local[(RadixKey(src_k[i]) >> shift) & 0xff]++;
          dst_k[to] = src_k[i];
          dst_v[to] = src_v[i];
        }
      } else {
        for (size_t i = lo; i < hi; i++) dst_k[local[(RadixKey(src_k[i]) >> shift) & 0xff]++] = src_k[i];
      }
    });
    std::swap(src_k, dst_k);
    std::swap(src_v, dst_v);
  }
  if (src_k != keys) {
    ParallelCopy(keys, src_k, n * sizeof(K));
    if (vals != nullptr) ParallelCopy(vals, src_v, n * sizeof(V));
  }
}

template <typename K, typename = std::enable_if_t<std::is_integral_v<K>>>
void sort(CpuPolicy&, K* first, K* last) {
  RadixSort<K, char>(first, nullptr, static_cast<size_t>(last - first));
}
template <typename K, typename V, typename = std::enable_if_t<std::is_integral_v<K>>>
void stable_sort_by_key(CpuPolicy&, K* keys_first, K* keys_last, V* values_first) {
  RadixSort<K, V>(keys_first, values_first, static_cast<size_t>(keys_last - keys_first));
}

}  // namespace xrasp
#endif  // XRASP_UTIL_CPU_PRIMITIVES_H
