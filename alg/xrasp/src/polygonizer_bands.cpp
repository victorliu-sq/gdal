#include "xrasp/core/polygonizer_bands.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "xrasp/steps/bands/b1_open_vertices.h"
#include "xrasp/steps/bands/b2_band_holes.h"
#include "xrasp/steps/s1_corners/base.h"
#include "xrasp/steps/s1_corners/opt_two_pass.h"
#include "xrasp/steps/s2_links/opt_line_index.h"
#include "xrasp/util/dev_value.h"
#include "xrasp/util/device_memory.h"
#include "xrasp/backend/launcher.h"

namespace xrasp {

namespace {

template <typename T>
DArray<T> Concatenate(std::vector<DArray<T>>* parts) {
  if (parts->size() == 1) {
    DArray<T> only = std::move((*parts)[0]);
    parts->clear();
    return only;
  }
  size_t total = 0;
  for (const auto& p : *parts) total += p.size();
  DArray<T> all(total);
  size_t at = 0;
  for (auto& p : *parts) {
    if (p.size() > 0) {
      CopyOnDevice(all.data() + at, p.data(), p.size() * sizeof(T));
    }
    at += p.size();
    p.Free();
  }
  parts->clear();
  return all;
}

template <typename T>
std::vector<T> ConcatenateHost(std::vector<std::vector<T>>* parts) {
  size_t total = 0;
  for (const auto& p : *parts) total += p.size();
  std::vector<T> all;
  all.reserve(total);
  for (auto& p : *parts) {
    all.insert(all.end(), p.begin(), p.end());
    std::vector<T>().swap(p);
  }
  parts->clear();
  return all;
}

// packed[i] = position * 4 + flag  ->  pos[i], flag[i]
void Unpack(DArray<int64_t>* pos, DArray<uint8_t>* flag) {
  flag->Reset(pos->size());
  auto p = pos->DeviceView();
  auto f = flag->DeviceView();
  ForEach(p.size(), [=] __device__(size_t i) {
    f[i] = static_cast<uint8_t>(p[i] & 3);
    p[i] >>= 2;
  });
}

// the rings of the first `count0` vertices (the down-entered ones), as global ring numbers
void RingsOfDownEntered(const Rings& rings, int64_t count0, int64_t ring_base, DArray<int64_t>* out) {
  out->Reset(count0);
  auto ring = rings.ring.DeviceView();
  auto o = out->DeviceView();
  ForEach(count0, [=] __device__(size_t i) { o[i] = ring_base + ring[i]; });
}

// parent[w] for the waiting holes: the ring of the first set-aside down-entered vertex at or after the key.
// Returns the number of waiting holes; `bad` is set when one found no such vertex in its column.
int64_t AnswerWaitingHoles(const DArray<int64_t>& pending, const DArray<int64_t>& pos, const DArray<int64_t>& ring,
                           int64_t hc, DArray<int64_t>* parent, bool* bad) {
  const int64_t nr = static_cast<int64_t>(pending.size());
  const int64_t n = static_cast<int64_t>(pos.size());
  DValue<int> waiting(0), wrong(0);
  auto w = pending.DeviceView();
  auto p = pos.DeviceView(), r = ring.DeviceView();
  auto par = parent->DeviceView();
  auto fw = waiting.DeviceView(), fb = wrong.DeviceView();
  ForEach(nr, [=] __device__(size_t i) {
    if (w[i] < 0) return;
    atomicAdd(fw.data(), 1);
    int64_t k = LowerBound(p.data(), n, w[i]);
    if (k < n && p[k] / hc == w[i] / hc) {
      par[i] = r[k];
    } else {
      fb.Set(1);
    }
  });
  *bad = wrong.GetD2H() != 0;
  return waiting.GetD2H();
}

// hole -> ring -> ... -> outer ring.  Returns the number of rounds.
int JumpToOuterRing(DArray<int64_t>* parent) {
  const int64_t nr = static_cast<int64_t>(parent->size());
  DArray<int64_t> next(nr);
  DValue<int> changed;
  int hops = 0;
  while (true) {
    changed.SetH2D(0);
    auto p = parent->DeviceView();
    auto q = next.DeviceView();
    auto flag = changed.DeviceView();
    ForEach(nr, [=] __device__(size_t r) {
      int64_t pp = p[p[r]];
      q[r] = pp;
      if (pp != p[r]) flag.Set(1);
    });
    if (changed.GetD2H() == 0) break;
    parent->Swap(next);
    hops++;
  }
  return hops;
}

struct IsFirstOfPolygon {
  const int64_t* key;      // sorted: outer ring * 2 + is_hole
  DEV_HOST_INLINE bool operator()(int64_t i) const { return i == 0 || (key[i] >> 1) != (key[i - 1] >> 1); }
};

// The rings ordered by outer ring, the outer ring first: `ord`, and where every polygon starts in it.
void OrderRingsOnDevice(const DArray<int64_t>& parent, const DArray<uint8_t>& is_hole, DArray<int64_t>* ord,
                        DArray<int64_t>* start) {
  const int64_t nr = static_cast<int64_t>(parent.size());
  DArray<int64_t> key(nr);
  ord->Reset(nr);
  {
    auto k = key.DeviceView();
    auto p = parent.DeviceView();
    auto h = is_hole.DeviceView();
    ForEach(nr, [=] __device__(size_t r) { k[r] = p[r] * 2 + h[r]; });
  }
  thrust::sequence(Exec(), ord->begin(), ord->end(), static_cast<int64_t>(0));
  thrust::stable_sort_by_key(Exec(), key.begin(), key.end(), ord->begin());
  start->Reset(nr);
  int64_t* end = thrust::copy_if(Exec(), thrust::make_counting_iterator<int64_t>(0),
                                 thrust::make_counting_iterator<int64_t>(nr), start->begin(),
                                 IsFirstOfPolygon{key.data()});
  start->Shrink(end - start->begin());
}

void OrderRings(const DArray<int64_t>& parent, const DArray<uint8_t>& is_hole, std::vector<int64_t>* order,
                std::vector<int64_t>* poly_start) {
  DArray<int64_t> ord, start;
  OrderRingsOnDevice(parent, is_hole, &ord, &start);
  ord.ToHost(order);
  start.ToHost(poly_start);
}

// ---- band_on_device: what the host does otherwise ------------------------------------------------------------
// The value of every ring: the value of one cell of its block, read in the raster in device memory.
template <typename T>
void RingValuesOfType(const void* cells_v, int64_t w, const Rings& rings, DArray<int64_t>* value) {
  value->Reset(rings.nr);
  const T* cells = static_cast<const T*>(cells_v);
  auto cx = rings.cell_x.DeviceView(), cy = rings.cell_y.DeviceView();
  auto out = value->DeviceView();
  ForEach(rings.nr, [=] __device__(size_t r) {
    out[r] = static_cast<int64_t>(cells[static_cast<int64_t>(cy[r]) * w + cx[r]]);
  });
}

void RingValues(CellType type, const void* cells, int64_t w, const Rings& rings, DArray<int64_t>* value) {
  switch (type) {
    case CellType::kU8: RingValuesOfType<uint8_t>(cells, w, rings, value); break;
    case CellType::kI8: RingValuesOfType<int8_t>(cells, w, rings, value); break;
    case CellType::kU16: RingValuesOfType<uint16_t>(cells, w, rings, value); break;
    case CellType::kI16: RingValuesOfType<int16_t>(cells, w, rings, value); break;
    case CellType::kU32: RingValuesOfType<uint32_t>(cells, w, rings, value); break;
    case CellType::kI32: RingValuesOfType<int32_t>(cells, w, rings, value); break;
  }
}

// parent[r] = ring_base + r, pending[r] = -1
void OwnParents(int64_t ring_base, int64_t nr, DArray<int64_t>* parent, DArray<int64_t>* pending) {
  parent->Reset(nr);
  pending->Reset(nr);
  auto p = parent->DeviceView(), w = pending->DeviceView();
  ForEach(nr, [=] __device__(size_t r) {
    p[r] = ring_base + static_cast<int64_t>(r);
    w[r] = -1;
  });
}

// The down-entered vertices of the rings finished in the bands, as the kernel below reads them: band b has
// n[b] sorted positions at pos[b] with their rings at ring[b], and starts at corner row y_first[b].
struct BandTables {
  DArray<int64_t> y_first, n;
  DArray<int64_t*> pos, ring;
};

// The holes of the rings that cross bands: west along the row from the hole's cell to the first grid line
// where the value changes, then the first down-entered vertex at or below in that column -- among the
// set-aside vertices and among those of the band of that row; the nearer one is the ring.
// False when a hole found none.
template <typename T>
bool AttachOpenHolesOfType(const void* cells_v, int64_t w, int64_t hc, const Rings& rings,
                           const DArray<int64_t>& open_pos, const DArray<int64_t>& open_ring,
                           const BandTables& bands, DArray<int64_t>* parent) {
  const T* cells = static_cast<const T*>(cells_v);
  const int64_t n_open = static_cast<int64_t>(open_pos.size());
  const int64_t n_bands = static_cast<int64_t>(bands.n.size());
  DValue<int> wrong(0);
  auto hs = rings.holes.DeviceView();
  auto cx = rings.hole_cx.DeviceView(), cy = rings.hole_cy.DeviceView();
  auto op = open_pos.DeviceView(), oring = open_ring.DeviceView();
  auto yf = bands.y_first.DeviceView(), bn = bands.n.DeviceView();
  auto bp = bands.pos.DeviceView(), br = bands.ring.DeviceView();
  auto par = parent->DeviceView();
  auto bad = wrong.DeviceView();
  ForEach(rings.nholes, [=] __device__(size_t i) {
    const T* row = cells + cy[i] * w;
    int64_t xh = 0;
    for (int64_t x = cx[i]; x >= 1; x--) {
      if (row[x] != row[x - 1]) {
        xh = x;
        break;
      }
    }
    const int64_t key = xh * hc + (cy[i] + 1);
    int64_t best = INT64_MAX, who = -1;
    int64_t k = LowerBound(op.data(), n_open, key);
    if (k < n_open && op[k] / hc == xh) {
      best = op[k];
      who = oring[k];
    }
    const int64_t b = UpperBound(yf.data(), n_bands, cy[i] + 1) - 1;      // the band of corner row cy + 1
    const int64_t* pos = bp[b];
    k = LowerBound(pos, bn[b], key);
    if (k < bn[b] && pos[k] / hc == xh && pos[k] < best) who = br[b][k];
    if (who < 0) {
      bad.Set(1);
    } else {
      par[hs[i]] = who;
    }
  });
  return wrong.GetD2H() == 0;
}

bool AttachOpenHoles(CellType type, const void* cells, int64_t w, int64_t hc, const Rings& rings,
                     const DArray<int64_t>& open_pos, const DArray<int64_t>& open_ring, const BandTables& bands,
                     DArray<int64_t>* parent) {
  switch (type) {
    case CellType::kU8: return AttachOpenHolesOfType<uint8_t>(cells, w, hc, rings, open_pos, open_ring, bands, parent);
    case CellType::kI8: return AttachOpenHolesOfType<int8_t>(cells, w, hc, rings, open_pos, open_ring, bands, parent);
    case CellType::kU16: return AttachOpenHolesOfType<uint16_t>(cells, w, hc, rings, open_pos, open_ring, bands, parent);
    case CellType::kI16: return AttachOpenHolesOfType<int16_t>(cells, w, hc, rings, open_pos, open_ring, bands, parent);
    case CellType::kU32: return AttachOpenHolesOfType<uint32_t>(cells, w, hc, rings, open_pos, open_ring, bands, parent);
    case CellType::kI32: return AttachOpenHolesOfType<int32_t>(cells, w, hc, rings, open_pos, open_ring, bands, parent);
  }
  return false;
}

// The tables of the result from the tables of the rings: the value and the first ring of every polygon, where
// the vertices of every ring start.
void ResultTables(const DArray<int32_t>& length, const DArray<int64_t>& value, const DArray<int64_t>& ord,
                  const DArray<int64_t>& start, DArray<int64_t>* values, DArray<int64_t>* poly_offsets,
                  DArray<int64_t>* ring_offsets) {
  const int64_t nr = static_cast<int64_t>(length.size()), np = static_cast<int64_t>(start.size());
  values->Reset(np);
  poly_offsets->Reset(np + 1);
  ring_offsets->Reset(nr + 1);
  {
    auto v = value.DeviceView(), o = ord.DeviceView(), s = start.DeviceView();
    auto out = values->DeviceView(), po = poly_offsets->DeviceView();
    ForEach(np + 1, [=] __device__(size_t p) {
      if (static_cast<int64_t>(p) == np) {
        po[p] = nr;
      } else {
        out[p] = v[o[s[p]]];
        po[p] = s[p];
      }
    });
  }
  {
    auto len = length.DeviceView();
    auto ro = ring_offsets->DeviceView();
    ForEach(nr + 1, [=] __device__(size_t r) { ro[r] = static_cast<int64_t>(r) < nr ? len[r] : 0; });
  }
  thrust::exclusive_scan(Exec(), ring_offsets->begin(), ring_offsets->end(), ring_offsets->begin());
}

struct IsSet {
  DEV_HOST_INLINE bool operator()(uint8_t v) const { return v != 0; }
};

// ---- a sink takes the complete polygons of a band (PolygonizerBands::EmitBand) ------------------------------
// root[r]: the ring the parents of r lead to, in the numbers of the band.  An outer ring leads to itself; so
// does a hole that waits for its ring (it is among the outlines that cross bands).
void LocalRoots(const DArray<int64_t>& parent, int64_t ring_base, DArray<int64_t>* root) {
  const int64_t nr = static_cast<int64_t>(parent.size());
  root->Reset(nr);
  {
    auto p = parent.DeviceView();
    auto q = root->DeviceView();
    ForEach(nr, [=] __device__(size_t r) { q[r] = p[r] - ring_base; });
  }
  JumpToOuterRing(root);
}

// offsets[r]: where the vertices of ring r start (one more at the end)
void RingOffsets(const DArray<int32_t>& length, DArray<int64_t>* offsets) {
  const int64_t nr = static_cast<int64_t>(length.size());
  offsets->Reset(nr + 1);
  auto len = length.DeviceView();
  auto o = offsets->DeviceView();
  ForEach(nr + 1, [=] __device__(size_t r) { o[r] = static_cast<int64_t>(r) < nr ? len[r] : 0; });
  thrust::exclusive_scan(Exec(), offsets->begin(), offsets->end(), offsets->begin());
}

// start with one more entry, `end`
void WithEnd(const DArray<int64_t>& start, int64_t end, DArray<int64_t>* out) {
  const int64_t n = static_cast<int64_t>(start.size());
  out->Reset(n + 1);
  auto s = start.DeviceView();
  auto o = out->DeviceView();
  ForEach(n + 1, [=] __device__(size_t i) { o[i] = static_cast<int64_t>(i) < n ? s[i] : end; });
}

// keep[r] = 1 for the rings that stay: those whose root is a hole.  number[r]: how many rings before r stay.
int64_t RingsThatStay(const DArray<int64_t>& root, const DArray<uint8_t>& is_hole, DArray<uint8_t>* keep,
                      DArray<int64_t>* number) {
  const int64_t nr = static_cast<int64_t>(root.size());
  keep->Reset(nr);
  number->Reset(nr);
  {
    auto q = root.DeviceView();
    auto h = is_hole.DeviceView();
    auto k = keep->DeviceView();
    auto n = number->DeviceView();
    ForEach(nr, [=] __device__(size_t r) {
      k[r] = h[q[r]];
      n[r] = h[q[r]];
    });
  }
  const int64_t count = static_cast<int64_t>(thrust::count_if(Exec(), keep->begin(), keep->end(), IsSet{}));
  thrust::exclusive_scan(Exec(), number->begin(), number->end(), number->begin());
  return count;
}

// out[number[r]] = in[r] for the rings that stay
template <typename T>
void TakeKept(const DArray<T>& in, const DArray<uint8_t>& keep, const DArray<int64_t>& number, int64_t count,
              DArray<T>* out) {
  out->Reset(count);
  auto a = in.DeviceView();
  auto k = keep.DeviceView();
  auto n = number.DeviceView();
  auto o = out->DeviceView();
  ForEach(in.size(), [=] __device__(size_t r) {
    if (k[r]) o[n[r]] = a[r];
  });
}

// out[number[r]] = offsets[r] for the rings that stay: where each of them starts among the vertices of the band
void TakeKeptStarts(const DArray<int64_t>& offsets, const DArray<uint8_t>& keep, const DArray<int64_t>& number,
                    int64_t count, DArray<int64_t>* out) {
  out->Reset(count);
  auto a = offsets.DeviceView();
  auto k = keep.DeviceView();
  auto n = number.DeviceView();
  auto o = out->DeviceView();
  ForEach(keep.size(), [=] __device__(size_t r) {
    if (k[r]) o[n[r]] = a[r];
  });
}

// The parents of the rings that stay, in the new numbers: a ring and its parent stay together.
void TakeKeptParents(const DArray<int64_t>& parent, int64_t ring_base, const DArray<uint8_t>& keep,
                     const DArray<int64_t>& number, int64_t count, DArray<int64_t>* out) {
  out->Reset(count);
  auto p = parent.DeviceView();
  auto k = keep.DeviceView();
  auto n = number.DeviceView();
  auto o = out->DeviceView();
  ForEach(parent.size(), [=] __device__(size_t r) {
    if (k[r]) o[n[r]] = ring_base + n[p[r] - ring_base];
  });
}

// The vertices of the rings that stay, ring after ring (new_offsets: of the kept rings).
void TakeKeptVertices(const DArray<int32_t>& x, const DArray<int32_t>& y, const DArray<int64_t>& offsets,
                      const DArray<uint8_t>& keep, const DArray<int64_t>& number,
                      const DArray<int64_t>& new_offsets, DArray<int32_t>* kx, DArray<int32_t>* ky) {
  const int64_t nr = static_cast<int64_t>(keep.size());
  const int64_t count = static_cast<int64_t>(new_offsets.size()) - 1;
  const int64_t total = count > 0 ? new_offsets.Get(count) : 0;
  kx->Reset(total);
  ky->Reset(total);
  auto vx = x.DeviceView(), vy = y.DeviceView();
  auto o = offsets.DeviceView(), no = new_offsets.DeviceView();
  auto k = keep.DeviceView();
  auto n = number.DeviceView();
  auto ox = kx->DeviceView(), oy = ky->DeviceView();
  ForEach(nr, [=] __device__(size_t r) {
    if (!k[r]) return;
    const int64_t from = o[r], to = no[n[r]], len = o[r + 1] - o[r];
    for (int64_t i = 0; i < len; i++) {
      ox[to + i] = vx[from + i];
      oy[to + i] = vy[from + i];
    }
  });
}

// The down-entered vertices of the rings that stay, with their rings in the new numbers.
void TakeKeptIndex(const DArray<int64_t>& pos, const DArray<int64_t>& ring, int64_t ring_base,
                   const DArray<uint8_t>& keep, const DArray<int64_t>& number, DArray<int64_t>* pos_out,
                   DArray<int64_t>* ring_out) {
  const int64_t n0 = static_cast<int64_t>(pos.size());
  DArray<uint8_t> stays(n0);
  DArray<int64_t> renumbered(n0);
  {
    auto r = ring.DeviceView();
    auto k = keep.DeviceView();
    auto n = number.DeviceView();
    auto s = stays.DeviceView();
    auto m = renumbered.DeviceView();
    ForEach(n0, [=] __device__(size_t i) {
      const int64_t local = r[i] - ring_base;
      s[i] = k[local];
      m[i] = ring_base + n[local];
    });
  }
  const int64_t count = static_cast<int64_t>(thrust::count_if(Exec(), stays.begin(), stays.end(), IsSet{}));
  pos_out->Reset(count);
  ring_out->Reset(count);
  thrust::copy_if(Exec(), pos.begin(), pos.end(), stays.data(), pos_out->begin(), IsSet{});
  thrust::copy_if(Exec(), renumbered.begin(), renumbered.end(), stays.data(), ring_out->begin(), IsSet{});
}

int64_t CountHoles(const DArray<uint8_t>& is_hole) {
  return static_cast<int64_t>(thrust::count_if(Exec(), is_hole.begin(), is_hole.end(), IsSet{}));
}

}  // namespace

PolygonizerBands::PolygonizerBands(const PolygonizeConfig& config, const Raster& raster)
    : config_(config), raster_(raster), h_(raster.h), w_(raster.w), wc_(raster.w + 1), hc_(raster.h + 1) {
  if (h_ < 1 || w_ < 1) throw std::runtime_error("the raster is empty");
  if (config_.labels) throw std::runtime_error("-labels needs -mode whole");
  rows_ = std::max<int64_t>(1, std::min<int64_t>(hc_, static_cast<int64_t>(config_.strip_cells) / std::max<int64_t>(1, w_)));
  counts_ = Json::object();
  // band_on_device: the raster is in device memory before anything is timed, a strip is rows of it
  if (config_.band_on_device) device_cells_.Load(raster_.cells<uint8_t>(), raster_.num_bytes());
}

// ---- s1 on one strip ----------------------------------------------------------------------------------------
template <typename T>
void PolygonizerBands::FindVerticesOfNextStripOfType() {
  const int64_t y0 = next_row_;
  const int64_t y1 = std::min(hc_, y0 + rows_);
  const int64_t c0 = std::max<int64_t>(y0 - 1, 0);
  const int64_t c1 = std::min(y1 - 1, h_ - 1);
  StripCells<T> strip;
  const T* strip_data = nullptr;
  if (config_.band_on_device) {
    strip_data = reinterpret_cast<const T*>(device_cells_.data()) + c0 * w_;
  } else {
    DeviceSync();
    auto t0 = std::chrono::steady_clock::now();
    strip.Load(raster_.cells<T>() + c0 * w_, static_cast<size_t>((c1 - c0 + 1) * w_));
    DeviceSync();
    upload_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    strip_data = strip.data();
  }
  DRasterView<T> view{strip_data, c0, c1 - c0 + 1, h_, w_};
  DArray<int64_t> pos[4], edges;
  DArray<uint8_t> flag[4];
  FindVerticesInStripAs(config_.corner_chunk, view, y0, y1 - y0, std::min(y1, h_) - y0, wc_, hc_, pos, flag,
                        &edges);
  for (int d = 0; d < 4; d++) {
    in_band_ += static_cast<int64_t>(pos[d].size());
    pos_parts_[d].push_back(std::move(pos[d]));
    flag_parts_[d].push_back(std::move(flag[d]));
  }
  edge_parts_.push_back(std::move(edges));
  next_row_ = y1;
  n_strips_++;
}

void PolygonizerBands::FindVerticesOfNextStrip() {
  switch (raster_.type) {
    case CellType::kU8: FindVerticesOfNextStripOfType<uint8_t>(); break;
    case CellType::kI8: FindVerticesOfNextStripOfType<int8_t>(); break;
    case CellType::kU16: FindVerticesOfNextStripOfType<uint16_t>(); break;
    case CellType::kI16: FindVerticesOfNextStripOfType<int16_t>(); break;
    case CellType::kU32: FindVerticesOfNextStripOfType<uint32_t>(); break;
    case CellType::kI32: FindVerticesOfNextStripOfType<int32_t>(); break;
  }
}

bool PolygonizerBands::BandIsComplete() const {
  return in_band_ >= static_cast<int64_t>(config_.band_vertices) || next_row_ >= hc_;
}

void PolygonizerBands::SortBand() {
  if (in_band_ > static_cast<int64_t>(INT32_MAX)) {
    throw std::runtime_error("a band holds more than 2^31 - 1 vertices: use a smaller -band_vertices or -strip_cells");
  }
  lists_ = VertexLists();
  lists_.wc = wc_;
  lists_.hc = hc_;
  for (int d : {1, 3}) {
    lists_.pos[d] = Concatenate(&pos_parts_[d]);
    lists_.flag[d] = Concatenate(&flag_parts_[d]);
  }
  for (int d : {0, 2}) {
    lists_.pos[d] = Concatenate(&pos_parts_[d]);
    flag_parts_[d].clear();
    SortAndUnpackVertical(&lists_.pos[d], &lists_.flag[d]);
  }
  lists_.SetCounts();
  edges_ = Concatenate(&edge_parts_);
  n_vertices_ += lists_.size();
}

// ---- b1 -----------------------------------------------------------------------------------------------------
void PolygonizerBands::SplitBand() {
  DArray<uint8_t> open;
  DArray<int32_t> links;                        // open_walk: the next vertex of every vertex inside the band
  DeviceSync();
  auto t0 = std::chrono::steady_clock::now();
  if (config_.link_index) BuildLineIndex(&lists_);
  FindOpenVertices(lists_, &open, config_.open_walk, &links);
  FreeLineIndex(&lists_);
  DeviceSync();
  split_open_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  t0 = std::chrono::steady_clock::now();
  VertexLists kept;
  kept.wc = wc_;
  kept.hc = hc_;
  DArray<int32_t> number;
  DArray<int64_t> packed_open[4];
  SplitLists(lists_, open, &number, packed_open, &kept, &owner_);
  for (int d = 0; d < 4; d++) {
    n_open_ += static_cast<int64_t>(packed_open[d].size());
    if (config_.band_on_device) {
      open_dev_[d].push_back(std::move(packed_open[d]));
    } else {
      open_[d].push_back(packed_open[d].ToHost());
      packed_open[d].Free();
    }
    if (d == 0) {
      pos0_all_ = std::move(lists_.pos[0]);     // all down-entered vertices of the band: the holes look here
    } else {
      lists_.pos[d].Free();
    }
    lists_.flag[d].Free();
  }
  kept.SetCounts();
  lists_ = std::move(kept);
  DeviceSync();
  split_groups_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  t0 = std::chrono::steady_clock::now();
  band_links_.Free();
  if (!links.empty()) KeptLinks(links, open, number, lists_.size(), &band_links_);    // step 2 of the band is done with this
  DeviceSync();
  split_links_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// ---- s2 .. s5 and b2 on the rings inside the band --------------------------------------------------------------
void PolygonizerBands::CloseBandRings() {
  band_has_rings_ = lists_.size() > 0;
  if (!band_has_rings_) return;
  pipe_ = RingPipeline();
  if (static_cast<int64_t>(band_links_.size()) == lists_.size()) {
    pipe_.UseLinks(std::move(band_links_), lists_);
  } else {
    if (config_.link_index) BuildLineIndex(&lists_);
    pipe_.Link(lists_);
    FreeLineIndex(&lists_);
  }
  band_links_.Free();
  pipe_.Rank(lists_, config_.tile, config_.rank_walk);
  pipe_.Repair(lists_, config_.tile);
  pipe_.Classify(lists_);
  n_one_block_ += pipe_.one_block;
  AttachBandHoles(pipe_.rings, edges_, pos0_all_, owner_, wc_, hc_, ring_base_, &parent_, &pending_);
}

// vertices in ring order and the ring tables of pipe_ -> chunk (parent and pending are filled by the caller)
void PolygonizerBands::RingsToChunk(Chunk* chunk) {
  Rings& rings = pipe_.rings;
  DArray<int32_t> xr, yr;
  VerticesInRingOrder(rings, pipe_.vx, pipe_.vy, &xr, &yr);
  pipe_.vx.Free();
  pipe_.vy.Free();
  chunk->n_rings = rings.nr;
  if (config_.band_on_device) {
    chunk->dx = std::move(xr);
    chunk->dy = std::move(yr);
    RingValues(raster_.type, device_cells_.data(), w_, rings, &chunk->d_value);
    chunk->d_len = std::move(rings.length);
    chunk->d_hole = std::move(rings.is_hole);
    return;
  }
  xr.ToHost(&chunk->x);
  yr.ToHost(&chunk->y);
  DeviceSync();
  auto t0 = std::chrono::steady_clock::now();
  rings.length.ToHost(&chunk->ring_len);
  rings.is_hole.ToHost(&chunk->is_hole);
  std::vector<int32_t> cx = rings.cell_x.ToHost(), cy = rings.cell_y.ToHost();
  chunk->value.resize(cx.size());
  for (size_t r = 0; r < cx.size(); r++) chunk->value[r] = raster_.Value(cx[r], cy[r]);
  DeviceSync();
  tables_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void PolygonizerBands::BandToHost() {
  BandIndex index;
  index.y_first = band_first_row_;
  if (band_has_rings_) {
    Chunk chunk;
    RingsToChunk(&chunk);
    DArray<int64_t> ring0;
    RingsOfDownEntered(pipe_.rings, lists_.count[0], ring_base_, &ring0);
    if (config_.band_on_device) {
      index.d_pos = std::move(lists_.pos[0]);
      index.d_ring = std::move(ring0);
      chunk.d_parent = std::move(parent_);
      chunk.d_pending = std::move(pending_);
    } else {
      DeviceSync();
      auto t0 = std::chrono::steady_clock::now();
      parent_.ToHost(&chunk.parent);
      pending_.ToHost(&chunk.pending);
      lists_.pos[0].ToHost(&index.pos);
      ring0.ToHost(&index.ring);
      DeviceSync();
      tables_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    if (config_.band_on_device && sink_) {
      // the arrays of the band go first: the sink writes while only the rings are left
      pipe_ = RingPipeline();
      lists_.Free();
      edges_.Free();
      pos0_all_.Free();
      owner_.Free();
      EmitBand(&chunk, &index);
    }
    ring_base_ += chunk.n_rings;
    if (chunk.n_rings > 0 || chunk.n_all > 0) chunks_.push_back(std::move(chunk));
  }
  band_index_.push_back(std::move(index));
  pipe_ = RingPipeline();
  lists_.Free();
  edges_.Free();
  pos0_all_.Free();
  owner_.Free();
  parent_.Free();
  pending_.Free();
  n_bands_++;
  in_band_ = 0;
  band_first_row_ = next_row_;
}

// band_on_device with a sink.  The polygons whose outer ring was finished in this band are complete: their
// holes lie inside that ring, so in this band.  The sink takes them, and they leave the chunk and the index.
// What stays is the holes that belong to a polygon whose outline crosses bands -- those that wait for their
// ring and those that lead to such a hole -- in ring numbers of their own.
void PolygonizerBands::EmitBand(Chunk* chunk, BandIndex* index) {
  const int64_t nr = chunk->n_rings;
  struct Timed {
    double* ms;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    ~Timed() { *ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
  } timed{&emit_ms_};
  DArray<int64_t> root, offsets, order, starts;
  LocalRoots(chunk->d_parent, ring_base_, &root);
  RingOffsets(chunk->d_len, &offsets);
  {
    DArray<int64_t> start;
    OrderRingsOnDevice(root, chunk->d_hole, &order, &start);
    WithEnd(start, nr, &starts);
  }
  const int64_t groups = static_cast<int64_t>(starts.size()) - 1;
  if (!sink_at_end_) {
    BandPolygons band;
    band.rings = nr;
    band.groups = groups;
    band.x = chunk->dx.data();
    band.y = chunk->dy.data();
    band.ring_offsets = offsets.data();
    band.order = order.data();
    band.group_start = starts.data();
    band.is_hole = chunk->d_hole.data();
    band.value = chunk->d_value.data();
    DeviceSync();
    auto t_sink = std::chrono::steady_clock::now();
    sink_(band);
    sink_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_sink).count();
  }
  DArray<uint8_t> keep;
  DArray<int64_t> number;
  const int64_t kept = RingsThatStay(root, chunk->d_hole, &keep, &number);
  root.Free();
  n_emitted_rings_ += nr - kept;
  if (kept == nr) return;                       // no polygon of this band is complete: it stays as it is
  Chunk rest;
  rest.n_rings = kept;
  TakeKept(chunk->d_len, keep, number, kept, &rest.d_len);
  TakeKept(chunk->d_hole, keep, number, kept, &rest.d_hole);
  TakeKept(chunk->d_value, keep, number, kept, &rest.d_value);
  TakeKept(chunk->d_pending, keep, number, kept, &rest.d_pending);
  TakeKeptParents(chunk->d_parent, ring_base_, keep, number, kept, &rest.d_parent);
  if (sink_at_end_) {
    // the band is drained later: its vertices stay where they are, the rings that stay are found in them
    TakeKeptStarts(offsets, keep, number, kept, &rest.d_start);
    rest.dx = std::move(chunk->dx);
    rest.dy = std::move(chunk->dy);
    rest.a_offsets = std::move(offsets);
    rest.a_order = std::move(order);
    rest.a_groups = std::move(starts);
    rest.a_hole = std::move(chunk->d_hole);
    rest.a_value = std::move(chunk->d_value);
    rest.n_all = nr;
    rest.n_groups = groups;
  } else {
    DArray<int64_t> new_offsets;
    RingOffsets(rest.d_len, &new_offsets);
    TakeKeptVertices(chunk->dx, chunk->dy, offsets, keep, number, new_offsets, &rest.dx, &rest.dy);
  }
  DArray<int64_t> pos, ring;
  TakeKeptIndex(index->d_pos, index->d_ring, ring_base_, keep, number, &pos, &ring);
  index->d_pos = std::move(pos);
  index->d_ring = std::move(ring);
  *chunk = std::move(rest);
}

// Sink at the end: the bands hand out their complete polygons one by one, and each is given back when the sink
// returns.  What the caller had of the polygons that cross bands goes first: it is not valid any more.
void PolygonizerBands::DrainBands() {
  r_values_.Free();
  r_poly_offsets_.Free();
  r_ring_index_.Free();
  r_ring_offsets_.Free();
  for (auto& c : chunks_) {
    if (c.n_all == 0) c = Chunk();
  }
  int64_t held = DeviceMemory::CurrentBytes();
  for (auto& c : chunks_) {
    if (c.n_all == 0) continue;
    BandPolygons band;
    band.rings = c.n_all;
    band.groups = c.n_groups;
    band.x = c.dx.data();
    band.y = c.dy.data();
    band.ring_offsets = c.a_offsets.data();
    band.order = c.a_order.data();
    band.group_start = c.a_groups.data();
    band.is_hole = c.a_hole.data();
    band.value = c.a_value.data();
    DeviceSync();
    auto t_sink = std::chrono::steady_clock::now();
    sink_(band);
    sink_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_sink).count();
    emit_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_sink).count();
    c = Chunk();
    // what was given back is not memory of the process any more: every quarter of a gigabyte (the allocator
    // walks its whole heap for this)
    if (held - DeviceMemory::CurrentBytes() >= (int64_t{1} << 28)) {
      DeviceMemory::ReturnFreed();
      held = DeviceMemory::CurrentBytes();
    }
  }
  DeviceMemory::ReturnFreed();
  chunks_.clear();
}

// ---- the outlines that left their band: all of them together ------------------------------------------------
void PolygonizerBands::CloseOpenRings() {
  if (n_open_ == 0) return;
  if (n_open_ > static_cast<int64_t>(INT32_MAX)) {
    throw std::runtime_error("more than 2^31 - 1 vertices are on rings that cross bands");
  }
  lists_ = VertexLists();
  lists_.wc = wc_;
  lists_.hc = hc_;
  for (int d = 0; d < 4; d++) {
    if (config_.band_on_device) {
      lists_.pos[d] = Concatenate(&open_dev_[d]);
    } else {
      std::vector<int64_t> packed = ConcatenateHost(&open_[d]);
      lists_.pos[d].FromHost(packed);
    }
    // horizontal groups: the bands come in row order, already sorted
    if (d == 0 || d == 2) thrust::sort(Exec(), lists_.pos[d].begin(), lists_.pos[d].end());
    Unpack(&lists_.pos[d], &lists_.flag[d]);
  }
  lists_.SetCounts();
  pipe_ = RingPipeline();
  if (config_.link_index) BuildLineIndex(&lists_);
  pipe_.Link(lists_);
  FreeLineIndex(&lists_);
  pipe_.Rank(lists_, config_.tile, config_.rank_walk);
  pipe_.Repair(lists_, config_.tile);
  pipe_.Classify(lists_);
  n_one_block_ += pipe_.one_block;
  const Rings& rings = pipe_.rings;
  if (config_.band_on_device) {
    Chunk chunk;
    DArray<int64_t> ring0;
    RingsOfDownEntered(rings, lists_.count[0], ring_base_, &ring0);
    OwnParents(ring_base_, rings.nr, &chunk.d_parent, &chunk.d_pending);
    if (rings.nholes > 0) {
      std::vector<int64_t> y_first, n;
      std::vector<int64_t*> pos, ring;
      for (auto& b : band_index_) {
        y_first.push_back(b.y_first);
        n.push_back(static_cast<int64_t>(b.d_pos.size()));
        pos.push_back(b.d_pos.data());
        ring.push_back(b.d_ring.data());
      }
      BandTables bands;
      bands.y_first.FromHost(y_first);
      bands.n.FromHost(n);
      bands.pos.FromHost(pos);
      bands.ring.FromHost(ring);
      if (!AttachOpenHoles(raster_.type, device_cells_.data(), w_, hc_, rings, lists_.pos[0], ring0, bands,
                           &chunk.d_parent)) {
        throw std::runtime_error("a hole found no ring of its block to the west");
      }
    }
    d_open_pos0_ = std::move(lists_.pos[0]);
    d_open_ring0_ = std::move(ring0);
    const int64_t nr_open = rings.nr;
    RingsToChunk(&chunk);
    ring_base_ += nr_open;
    chunks_.push_back(std::move(chunk));
    pipe_ = RingPipeline();
    lists_.Free();
    return;
  }

  lists_.pos[0].ToHost(&open_pos0_);
  {
    DArray<int64_t> ring0;
    RingsOfDownEntered(rings, lists_.count[0], ring_base_, &ring0);
    ring0.ToHost(&open_ring0_);
  }
  Chunk chunk;
  chunk.parent.resize(rings.nr);
  for (int64_t r = 0; r < rings.nr; r++) chunk.parent[r] = ring_base_ + r;
  chunk.pending.assign(rings.nr, -1);

  // These holes may sit next to a ring that was finished in a band: look in both places, on the host.
  if (rings.nholes > 0) {
    std::vector<int32_t> holes = rings.holes.ToHost();
    std::vector<int64_t> cx = rings.hole_cx.ToHost(), cy = rings.hole_cy.ToHost();
    for (size_t i = 0; i < holes.size(); i++) {
      // west along the row from the cell to the first grid line where the value changes (0: the raster's edge)
      int64_t xh = 0;
      for (int64_t x = cx[i]; x >= 1; x--) {
        if (raster_.Value(x, cy[i]) != raster_.Value(x - 1, cy[i])) {
          xh = x;
          break;
        }
      }
      const int64_t key = xh * hc_ + (cy[i] + 1);
      int64_t best = INT64_MAX, who = -1;
      {
        auto it = std::lower_bound(open_pos0_.begin(), open_pos0_.end(), key);
        if (it != open_pos0_.end() && *it / hc_ == xh) {
          best = *it;
          who = open_ring0_[it - open_pos0_.begin()];
        }
      }
      // the band of corner row cy + 1
      size_t b = 0;
      while (b + 1 < band_index_.size() && band_index_[b + 1].y_first <= cy[i] + 1) b++;
      const BandIndex& band = band_index_[b];
      auto it = std::lower_bound(band.pos.begin(), band.pos.end(), key);
      if (it != band.pos.end() && *it / hc_ == xh && *it < best) {
        who = band.ring[it - band.pos.begin()];
      }
      if (who < 0) throw std::runtime_error("a hole found no ring of its block to the west");
      chunk.parent[holes[i]] = who;
    }
  }
  RingsToChunk(&chunk);
  ring_base_ += rings.nr;
  chunks_.push_back(std::move(chunk));
  pipe_ = RingPipeline();
  lists_.Free();
}

// ---- all rings: holes that were waiting, then polygons ------------------------------------------------------
void PolygonizerBands::MergeRings() {
  if (config_.band_on_device) {
    MergeRingsOnDevice();
    return;
  }
  const int64_t nr = ring_base_;
  std::vector<int32_t> rlen;
  std::vector<uint8_t> is_hole;
  std::vector<int64_t> parent, pending, value;
  rlen.reserve(nr);
  is_hole.reserve(nr);
  parent.reserve(nr);
  pending.reserve(nr);
  value.reserve(nr);
  for (auto& c : chunks_) {
    rlen.insert(rlen.end(), c.ring_len.begin(), c.ring_len.end());
    is_hole.insert(is_hole.end(), c.is_hole.begin(), c.is_hole.end());
    parent.insert(parent.end(), c.parent.begin(), c.parent.end());
    pending.insert(pending.end(), c.pending.begin(), c.pending.end());
    value.insert(value.end(), c.value.begin(), c.value.end());
    std::vector<int32_t>().swap(c.ring_len);
    std::vector<uint8_t>().swap(c.is_hole);
    std::vector<int64_t>().swap(c.parent);
    std::vector<int64_t>().swap(c.pending);
    std::vector<int64_t>().swap(c.value);
  }
  // the ring tables are small next to the vertices: they go back to the GPU for this
  int64_t waiting = 0;
  int hops = 0;
  int64_t nholes = 0;
  for (uint8_t f : is_hole) nholes += f;
  std::vector<int64_t> order, poly_start;
  {
    DArray<int64_t> d_parent, d_pending, d_pos, d_ring;
    DArray<uint8_t> d_hole;
    d_parent.FromHost(parent);
    d_pending.FromHost(pending);
    d_hole.FromHost(is_hole);
    d_pos.FromHost(open_pos0_);
    d_ring.FromHost(open_ring0_);
    bool bad = false;
    waiting = AnswerWaitingHoles(d_pending, d_pos, d_ring, hc_, &d_parent, &bad);
    if (bad) throw std::runtime_error("a waiting hole found no set-aside vertex below it");
    hops = JumpToOuterRing(&d_parent);
    OrderRings(d_parent, d_hole, &order, &poly_start);
  }
  const int64_t npoly = static_cast<int64_t>(poly_start.size());
  out_.poly_offsets = poly_start;
  out_.poly_offsets.push_back(nr);
  out_.values.resize(npoly);
  for (int64_t p = 0; p < npoly; p++) out_.values[p] = value[order[poly_start[p]]];
  out_.ring_index = std::move(order);
  out_.ring_offsets.resize(nr + 1);
  out_.ring_offsets[0] = 0;
  for (int64_t r = 0; r < nr; r++) out_.ring_offsets[r + 1] = out_.ring_offsets[r] + rlen[r];
  const int64_t total = out_.ring_offsets[nr];
  out_.x.Reset(total);
  out_.y.Reset(total);
  int64_t at = 0;
  DeviceSync();
  auto t0 = std::chrono::steady_clock::now();
  for (auto& c : chunks_) {                     // each piece is released as soon as it is copied
    CopyOnHost(out_.x.data() + at, c.x.data(), c.x.size() * sizeof(int32_t));
    CopyOnHost(out_.y.data() + at, c.y.data(), c.y.size() * sizeof(int32_t));
    at += static_cast<int64_t>(c.x.size());
    c.x.Free();
    c.y.Free();
  }
  DeviceSync();
  const double gather_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  chunks_.clear();

  counts_["Bands"] = n_bands_;
  counts_["Strips"] = n_strips_;
  counts_["StripRows"] = rows_;
  counts_["UploadTime"] = upload_ms_;
  counts_["SplitOpenTime"] = split_open_ms_;       // SplitBand: the links inside the band, which vertices are open
  counts_["SplitGroupsTime"] = split_groups_ms_;   // SplitBand: the open vertices and the others apart
  counts_["SplitLinksTime"] = split_links_ms_;     // SplitBand: the links of the vertices that stay
  counts_["TablesToHostTime"] = tables_ms_;   // the tables of the rings of every band to host memory, their values
  counts_["GatherVerticesTime"] = gather_ms;   // the vertices of all pieces into one array in host memory
  counts_["Vertices"] = n_vertices_;
  counts_["OpenVertices"] = n_open_;
  counts_["DiagonalPairsOneBlock"] = n_one_block_;
  counts_["Rings"] = nr;
  counts_["Holes"] = nholes;
  counts_["HolesWaiting"] = waiting;
  counts_["HoleJumpRounds"] = hops;
  counts_["Polygons"] = npoly;
  counts_["OuterRings"] = nr - nholes;
}

// band_on_device: the same in device memory.  The vertices stay where the bands left them, one piece per band
// and one of the rings that cross bands; the tables of the result index them as if they were one array.
void PolygonizerBands::MergeRingsOnDevice() {
  const int64_t nr = ring_base_;
  for (auto& b : band_index_) {
    b.d_pos.Free();
    b.d_ring.Free();
  }
  std::vector<DArray<int32_t>> len_parts;
  std::vector<DArray<uint8_t>> hole_parts;
  std::vector<DArray<int64_t>> parent_parts, pending_parts, value_parts;
  for (auto& c : chunks_) {
    len_parts.push_back(std::move(c.d_len));
    hole_parts.push_back(std::move(c.d_hole));
    parent_parts.push_back(std::move(c.d_parent));
    pending_parts.push_back(std::move(c.d_pending));
    value_parts.push_back(std::move(c.d_value));
  }
  DArray<int32_t> length = Concatenate(&len_parts);
  DArray<uint8_t> is_hole = Concatenate(&hole_parts);
  DArray<int64_t> parent = Concatenate(&parent_parts);
  DArray<int64_t> pending = Concatenate(&pending_parts);
  DArray<int64_t> value = Concatenate(&value_parts);
  bool bad = false;
  const int64_t waiting = AnswerWaitingHoles(pending, d_open_pos0_, d_open_ring0_, hc_, &parent, &bad);
  if (bad) throw std::runtime_error("a waiting hole found no set-aside vertex below it");
  pending.Free();
  d_open_pos0_.Free();
  d_open_ring0_.Free();
  const int hops = JumpToOuterRing(&parent);
  const int64_t nholes = CountHoles(is_hole);
  DArray<int64_t> start;
  OrderRingsOnDevice(parent, is_hole, &r_ring_index_, &start);
  parent.Free();
  is_hole.Free();
  const int64_t npoly = static_cast<int64_t>(start.size());
  ResultTables(length, value, r_ring_index_, start, &r_values_, &r_poly_offsets_, &r_ring_offsets_);
  length.Free();
  value.Free();
  start.Free();

  counts_["Bands"] = n_bands_;
  counts_["Strips"] = n_strips_;
  counts_["StripRows"] = rows_;
  counts_["UploadTime"] = upload_ms_;
  counts_["SplitOpenTime"] = split_open_ms_;       // SplitBand: the links inside the band, which vertices are open
  counts_["SplitGroupsTime"] = split_groups_ms_;   // SplitBand: the open vertices and the others apart
  counts_["SplitLinksTime"] = split_links_ms_;     // SplitBand: the links of the vertices that stay
  counts_["Vertices"] = n_vertices_;
  counts_["OpenVertices"] = n_open_;
  counts_["DiagonalPairsOneBlock"] = n_one_block_;
  counts_["Rings"] = nr;
  counts_["Holes"] = nholes;
  counts_["HolesWaiting"] = waiting;
  counts_["HoleJumpRounds"] = hops;
  counts_["Polygons"] = npoly;
  counts_["OuterRings"] = nr - nholes;
  counts_["ResultDeviceBytes"] = DeviceMemory::CurrentBytes();    // the raster and the result
  if (sink_) DeviceMemory::ReturnFreed();      // the arrays of the last steps, before the caller writes
  if (sink_) counts_["RingsGivenToSink"] = n_emitted_rings_;       // rings of polygons a sink took band by band
  counts_["RasterDeviceBytes"] = kKernelsReadHostMemory ? 0 : static_cast<int64_t>(raster_.num_bytes());
}

// band_on_device: the result copied to host memory (to check it or to write it; nothing of a run's time)
void PolygonizerBands::ResultToHost() {
  if (!config_.band_on_device || result_on_host_) return;
  result_on_host_ = true;
  r_values_.ToHost(&out_.values);
  r_poly_offsets_.ToHost(&out_.poly_offsets);
  r_ring_index_.ToHost(&out_.ring_index);
  r_ring_offsets_.ToHost(&out_.ring_offsets);
  const int64_t total = out_.ring_offsets.back();
  out_.x.Reset(total);
  out_.y.Reset(total);
  int64_t at = 0;
  for (auto& c : chunks_) {
    CopyToHost(out_.x.data() + at, c.dx.data(), c.dx.size() * sizeof(int32_t));
    CopyToHost(out_.y.data() + at, c.dy.data(), c.dy.size() * sizeof(int32_t));
    at += static_cast<int64_t>(c.dx.size());
  }
}

}  // namespace xrasp
