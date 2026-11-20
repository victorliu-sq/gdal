#ifndef XRASP_CORE_POLYGONIZE_WHOLE_H
#define XRASP_CORE_POLYGONIZE_WHOLE_H
// The whole procedure as one function (PolygonizeWhole) or in two parts (WholePolygonizer), for a host
// application (X-Rasp inside GDAL): the raster is in device memory at once (mode = whole, no block labels),
// and the steps run one after the other without the timers, the json and the configuration of the programs.
// core/polygonizer_strips does the same step by step for xrasp_polygonize; the unit tests compare the two.
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "xrasp/core/ring_pipeline.h"
#include "xrasp/steps/s1_corners/base.h"
#include "xrasp/steps/s1_corners/opt_two_pass.h"
#include "xrasp/steps/s2_links/opt_line_index.h"
#include "xrasp/steps/s6_holes/by_west_search.h"
#include "xrasp/steps/s7_assemble/base.h"
#include "xrasp/structs/dev_raster.h"
#include "xrasp/structs/polygons.h"
#include "xrasp/structs/vertex_lists.h"
#include "xrasp/util/dev_array.h"
#include "xrasp/backend/launcher.h"

#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace xrasp {

struct WholeOptions {
  int corner_chunk = -1;        // s1 in two passes, a thread taking the corners of 32 bytes of cells (0: base)
  bool link_index = true;       // s2: the next vertex searched in its own row or column
  int tile = 256;               // s3, s4: two-level ranking with tiles of this many corners (0: flat)
  bool rank_walk = false;       // s3: level 1 of the two-level ranking by walking (what a CPU wants)
  bool gdal_order = false;      // s7: polygons, holes and vertices in the order GDAL's polygonizer writes them
};

struct WholeCounts {
  int64_t vertices = 0, rings = 0, holes = 0, polygons = 0;
};

// Where the time and the memory of one call go: for every step its name, its milliseconds, and the bytes in
// live arrays when it ends; and the most bytes there were at any moment of the call.
struct WholeProfile {
  struct Step {
    std::string name;
    double ms;
    int64_t bytes;
  };
  std::vector<Step> steps;
  int64_t peak_bytes = 0;
};

// Marks the end of a step in a profile.
class WholeClock {
 public:
  explicit WholeClock(WholeProfile* profile) : profile_(profile), clock_(std::chrono::steady_clock::now()) {}
  void operator()(const char* name) {
    if (profile_ == nullptr) return;
    DeviceSync();
    auto now = std::chrono::steady_clock::now();
    profile_->steps.push_back({name, std::chrono::duration<double, std::milli>(now - clock_).count(),
                               DeviceMemory::CurrentBytes()});
    profile_->peak_bytes = DeviceMemory::PeakBytes();
    clock_ = now;
  }

 private:
  WholeProfile* profile_;
  std::chrono::steady_clock::time_point clock_;
};

// The procedure in two parts, for a host that has work of its own to do with the polygons while their order
// is found (GDAL makes its feature objects on one thread, which takes as long as the order on all the others):
//
//   FindPolygons   steps 1 to 7.  The polygons are GDAL's, the holes of a polygon and the vertices of a ring
//                  in GDAL's order, and the polygons in the order of the row of their last cell -- GDAL's
//                  order but for the polygons that end in the same row.
//   FindOrder      order[k]: the polygon GDAL writes k-th.  It reads nothing of the polygons, so the caller
//                  may run it on another thread while it reads them.
//
// PolygonizeWhole with options.gdal_order is both at once, the polygons stored in GDAL's order.
template <typename T>
class WholePolygonizer {
 public:
  // cells: h rows of w cells in device memory (host memory on the CPU); they must stay until
  // FindOrder is done.  options.gdal_order is taken as set.
  void FindPolygons(const T* cells, int64_t h, int64_t w, const WholeOptions& options, Polygons* out,
                    WholeCounts* counts = nullptr, WholeProfile* profile = nullptr) {
    if (profile != nullptr) DeviceMemory::ResetPeak();
    WholeClock done(profile);
    cells_ = cells;
    w_ = w;
    ToHoles(cells, h, w, options, /* keep_links = */ true, &done);
    FindRingStarts(lists_.wc, pipe_.vx, pipe_.vy, pipe_.rings, &scan_);
    done("starts");
    Assemble(out, counts, &scan_, &outer_);
    done("assemble");
    // what the order still reads: the links and the y of the vertices, the ring of the vertices entered
    // moving down and their positions, the outer ring of every ring, the edges
    pipe_.rings.rank.Free();
    pipe_.rings.length.Free();
    scan_.first.Free();
    scan_.first_rank.Free();
  }

  void FindOrder(DArray<int32_t>* order, WholeProfile* profile = nullptr) {
    WholeClock done(profile);
    // what the numbers need of the links; then the links, the corners of the vertices and the rings of all
    // but the vertices entered moving down go
    DArray<int32_t> top_y, from;
    FindEdgeTops(lists_, pipe_.nxt, pipe_.vy, &top_y, &from);
    pipe_.nxt.Free();
    pipe_.vx.Free();
    pipe_.vy.Free();
    pipe_.rings.ring.Shrink(static_cast<size_t>(lists_.count[0]));
    done("tops");
    FindPolygonNumbers(cells_, w_, lists_, edges_, top_y, from, pipe_.rings, &scan_);
    top_y.Free();
    from.Free();
    done("numbers");
    const int64_t np = static_cast<int64_t>(outer_.size());
    DArray<int64_t> key(np);
    order->Reset(np);
    thrust::sequence(Exec(), order->begin(), order->end(), 0);
    {
      // two stable sorts: the polygon's number, then its last row (the polygons are in that order already)
      auto k = key.DeviceView();
      auto ord = order->DeviceView();
      auto o = outer_.DeviceView();
      auto seed = scan_.seed.DeviceView(), last_row = scan_.last_row.DeviceView();
      ForEach(np, [=] __device__(size_t p) { k[p] = seed[o[p]]; });
      thrust::stable_sort_by_key(Exec(), key.begin(), key.end(), order->begin());
      ForEach(np, [=] __device__(size_t i) { k[i] = last_row[o[ord[i]]]; });
      thrust::stable_sort_by_key(Exec(), key.begin(), key.end(), order->begin());
    }
    done("sort");
    pipe_ = RingPipeline();
    lists_.Free();
    edges_.Free();
    scan_ = ScanOrder();
    outer_.Free();
  }

  // Steps 1 to 6 (PolygonizeWhole uses them too).
  void ToHoles(const T* cells, int64_t h, int64_t w, const WholeOptions& options, bool keep_links,
               WholeClock* clock) {
    ToLinks(cells, h, w, options, clock);
    WholeClock& done = *clock;
    if (options.tile > 0) {                                   // s3
      pipe_.RankAndRelease(&lists_, options.tile, options.rank_walk);
    } else {
      pipe_.Rank(lists_, 0);
    }
    done("rank");
    pipe_.Repair(lists_, options.tile);                       // s4
    done("repair");
    pipe_.Classify(lists_, keep_links);                       // s5
    done("rings");
    AttachHolesByWestSearch(lists_, edges_, &pipe_.rings);    // s6
    done("holes");
    // s7, first half: the value of a ring is the value of one cell of its block
    Rings& rings = pipe_.rings;
    ring_value_.Reset(rings.nr);
    auto cx = rings.cell_x.DeviceView(), cy = rings.cell_y.DeviceView();
    auto value = ring_value_.DeviceView();
    ForEach(rings.nr, [=] __device__(size_t r) {
      value[r] = static_cast<int64_t>(cells[static_cast<int64_t>(cy[r]) * w + cx[r]]);
    });
  }

  // Steps 1 and 2.  (Virtual: core/polygonize_whole_bitmap.h has them on bitmaps.)
  virtual void ToLinks(const T* cells, int64_t h, int64_t w, const WholeOptions& options, WholeClock* clock) {
    if (h < 1 || w < 1) throw std::runtime_error("the raster is empty");
    const int64_t wc = w + 1, hc = h + 1;
    WholeClock& done = *clock;
    // s1: one strip with every corner row
    DRasterView<T> view{cells, 0, h, h, w};
    DArray<int64_t> pos[4];
    DArray<uint8_t> flag[4];
    FindVerticesInStripAs(options.corner_chunk, view, 0, hc, h, wc, hc, pos, flag, &edges_);
    lists_.wc = wc;
    lists_.hc = hc;
    for (int d : {1, 3}) {
      lists_.pos[d] = std::move(pos[d]);
      lists_.flag[d] = std::move(flag[d]);
    }
    for (int d : {0, 2}) {
      lists_.pos[d] = std::move(pos[d]);
    }
    done("vertices");
    for (int d : {0, 2}) SortAndUnpackVertical(&lists_.pos[d], &lists_.flag[d]);
    lists_.SetCounts();
    done("sort");
    if (lists_.size() > static_cast<int64_t>(INT32_MAX)) {
      throw std::runtime_error("more than 2^31 - 1 vertices: this needs the band version");
    }
    if (options.link_index) BuildLineIndex(&lists_);          // s2
    pipe_.Link(lists_);
    FreeLineIndex(&lists_);
    done("link");
  }
  virtual ~WholePolygonizer() = default;

  // Step 7.  scan: null, or what is known of GDAL's order.
  void Assemble(Polygons* out, WholeCounts* counts, const ScanOrder* scan, DArray<int32_t>* outer) {
    if (counts != nullptr) {
      counts->rings = pipe_.rings.nr;
      counts->holes = pipe_.rings.nholes;
    }
    AssemblePolygons(pipe_.rings, ring_value_, pipe_.vx, pipe_.vy, out, scan, outer);
    ring_value_.Free();
    if (counts != nullptr) {
      counts->vertices = out->num_vertices();
      counts->polygons = out->num_polygons();
    }
  }

  const T* cells_ = nullptr;
  int64_t w_ = 0;
  VertexLists lists_;
  DArray<int64_t> edges_;
  RingPipeline pipe_;
  DArray<int64_t> ring_value_;
  ScanOrder scan_;
  DArray<int32_t> outer_;       // the outer ring of every polygon of FindPolygons
};

// Steps 1 to 7 with the polygonizer given (its steps 1 and 2 may be its own).
template <typename T>
void PolygonizeWholeWith(WholePolygonizer<T>* run, const T* cells, int64_t h, int64_t w,
                         const WholeOptions& options, Polygons* out, WholeCounts* counts, WholeClock* clock) {
  WholeClock& done = *clock;
  run->ToHoles(cells, h, w, options, options.gdal_order, &done);
  ScanOrder scan;
  if (options.gdal_order) {
    FindScanOrder(cells, w, run->lists_, run->edges_, run->pipe_.nxt, run->pipe_.vx, run->pipe_.vy,
                  run->pipe_.rings, &scan);
    done("order");
  }
  run->pipe_.nxt.Free();
  run->edges_.Free();
  run->lists_.Free();
  run->Assemble(out, counts, options.gdal_order ? &scan : nullptr, nullptr);
  done("assemble");
}

// cells: h rows of w cells in device memory (host memory on the CPU).  The polygons are GDAL's; with
// options.gdal_order they also come in GDAL's order (steps/s7_assemble/opt_scan_order.h).
template <typename T>
void PolygonizeWhole(const T* cells, int64_t h, int64_t w, const WholeOptions& options, Polygons* out,
                     WholeCounts* counts = nullptr, WholeProfile* profile = nullptr) {
  if (profile != nullptr) DeviceMemory::ResetPeak();
  WholeClock done(profile);
  WholePolygonizer<T> run;
  PolygonizeWholeWith(&run, cells, h, w, options, out, counts, &done);
}

}  // namespace xrasp
#endif  // XRASP_CORE_POLYGONIZE_WHOLE_H
