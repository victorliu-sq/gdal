#ifndef XRASP_STEPS_S1_CORNERS_OPT_TWO_PASS_H
#define XRASP_STEPS_S1_CORNERS_OPT_TWO_PASS_H
// Optimization of step 1: find the vertices in two passes over the strip.
//
// s1_corners/base.h goes over every corner of the strip about eleven times: once to write a byte per corner,
// then for each of the four directions once to count its vertices and once to copy them out, and twice more
// over the raster for the vertical unit edges.  Where the polygons are large almost all of those reads and
// writes are of corners at which nothing happens (zones-30s: 1 corner in 100 is a vertex), and the step is
// most of the run.
//
// Here a GPU thread takes `chunk` consecutive corners of one corner row (a chunk never crosses a row), and
// kChunksPerBlock(chunk) consecutive chunks are a block:
//
//   pass 1   a thread walks its chunk and counts the vertices of each direction and the vertical unit edges:
//            five counts of 12 bits in one 64-bit word per chunk;
//   offsets  a thread per block turns the counts of its chunks into where each chunk writes within the block
//            (again five numbers of 12 bits; one more bit says whether the chunk has anything to write) and
//            writes the five totals of the block; a prefix sum over the blocks, per list, says where every
//            block writes;
//   pass 2   a thread walks its chunk again and writes the vertices and the edges in place.  A chunk with
//            nothing to write returns without reading the raster.
//
// Walking a chunk, a thread carries the two cells left of the corner over from the corner before, so it reads
// two cells per corner instead of four, and it does no 64-bit division per corner.  Small chunks take more
// memory (8 bytes per chunk, against one byte per corner in base.h) and more time for the offsets; large
// ones make a thread read more cells one after the other.  Fastest on an L4 is the chunk whose cells of one
// row are 32 bytes (2026-11-17, step 1 without the upload, base.h -> two passes: zones-30s, 16-bit cells,
// 68.5 -> 15.5 ms with 16 corners, 24.2 ms with 32; gsw, 8-bit cells, 148.6 -> 25.0 ms with 32 corners,
// 56.5 ms with 64; corine, 8-bit, 367.3 -> 115.3 ms with 32, of which 63 ms sort the vertical vertices).
// chunk = -1 asks for that size.
//
// The output is that of FindVerticesInStrip, element for element: chunks are in corner order and a thread
// writes its chunk in order, so the horizontal lists and the edges come out sorted.
#include <chrono>
#include <cstdint>
#include <stdexcept>

#include <thrust/scan.h>

#include "xrasp/steps/s1_corners/base.h"

namespace xrasp {

constexpr int64_t kMaxCornerChunk = 255;
constexpr int kCountBits = 12;                // a count or an offset within a block: 0 .. 4095
constexpr uint64_t kCountMask = (1ull << kCountBits) - 1;
constexpr uint64_t kChunkWrites = 1ull << 63;

// Chunks per block: as many as keep the corners of a block within 12 bits, at most 32.
inline int64_t ChunksPerBlock(int64_t chunk) {
  int64_t fit = static_cast<int64_t>(kCountMask) / chunk;
  return fit < 1 ? 1 : (fit > 32 ? 32 : fit);
}

// Where step 1 spent its time (milliseconds), summed over the strips.
struct TwoPassTimes {
  double count = 0, offsets = 0, write = 0;
};

// Calls visit(x, code, edge) for the corners x0 .. x1 - 1 of corner row y at which the four cells are not all
// equal: `code` as CornerCode gives it, `edge` whether cells (x-1, y) and (x, y) differ.
template <typename T, typename Turn, typename Visit>
DEV_INLINE void WalkChunk(const DRasterView<T>& r, int64_t y, int64_t x0, int64_t x1, Visit& visit) {
  const T* up = y - 1 >= 0 ? r.data + (y - 1 - r.row0) * r.w : nullptr;       // cell row y - 1
  const T* low = y < r.h ? r.data + (y - r.row0) * r.w : nullptr;             // cell row y
  int64_t ul = (up != nullptr && x0 > 0) ? static_cast<int64_t>(up[x0 - 1]) : kOutside;
  int64_t ll = (low != nullptr && x0 > 0) ? static_cast<int64_t>(low[x0 - 1]) : kOutside;
  for (int64_t x = x0; x < x1; x++) {
    int64_t ur = (up != nullptr && x < r.w) ? static_cast<int64_t>(up[x]) : kOutside;
    int64_t lr = (low != nullptr && x < r.w) ? static_cast<int64_t>(low[x]) : kOutside;
    if (ul != ur || ll != lr || ul != ll) {
      visit(x, CornerCodeOf<Turn>(ul, ur, ll, lr), ll != lr);
    }
    ul = ur;
    ll = lr;
  }
}

struct CountInChunk {
  uint64_t packed = 0;      // five counts of kCountBits bits: directions 0 .. 3, vertical unit edges
  bool want_edges;
  DEV_INLINE void operator()(int64_t, uint8_t code, bool edge) {
    packed += static_cast<uint64_t>((code & 3) != 0) |
              (static_cast<uint64_t>(((code >> 2) & 3) != 0) << kCountBits) |
              (static_cast<uint64_t>(((code >> 4) & 3) != 0) << (2 * kCountBits)) |
              (static_cast<uint64_t>(((code >> 6) & 3) != 0) << (3 * kCountBits)) |
              (static_cast<uint64_t>(edge && want_edges) << (4 * kCountBits));
  }
};

struct WriteChunk {
  int64_t* pos[4];
  uint8_t* flag[4];         // of the horizontal directions; the vertical ones are packed into pos
  int64_t* edges;
  int64_t at[5];            // where this chunk writes next
  int64_t y, wc, hc;
  bool want_edges;
  DEV_INLINE void operator()(int64_t x, uint8_t code, bool edge) {
    for (int d = 0; d < 4; d++) {
      uint8_t state = (code >> (2 * d)) & 3;
      if (state == 0) continue;
      uint8_t f = state == 1 ? kFlagRight : (state == 3 ? kFlagDiagonal : 0);
      if (d == 1 || d == 3) {
        pos[d][at[d]] = y * wc + x;
        flag[d][at[d]] = f;
      } else {
        pos[d][at[d]] = (x * hc + y) * 4 + f;
      }
      at[d]++;
    }
    if (edge && want_edges) edges[at[4]++] = y * wc + x;
  }
};

// The same arguments and the same output as FindVerticesInStrip (s1_corners/base.h); chunk: corners per GPU
// thread, 1 .. kMaxCornerChunk.  times != nullptr: the time of each part is added to it (the device is
// synchronized between the parts).
template <typename T, typename Turn = TurnByValue>
void FindVerticesInStripTwoPass(const DRasterView<T>& raster, int64_t y0, int64_t n_rows, int64_t n_edge_rows,
                                int64_t wc, int64_t hc, int64_t chunk, DArray<int64_t> pos[4],
                                DArray<uint8_t> flag[4], DArray<int64_t>* edges, TwoPassTimes* times = nullptr) {
  if (chunk < 1 || chunk > kMaxCornerChunk) throw std::runtime_error("the corner chunk must be 1 .. 255");
  const int64_t per_row = (wc + chunk - 1) / chunk;          // chunks per corner row
  const int64_t n_chunks = n_rows * per_row;
  const int64_t per_block = ChunksPerBlock(chunk);
  const int64_t n_blocks = (n_chunks + per_block - 1) / per_block;
  const bool want_edges = edges != nullptr;
  auto now = [&]() {
    if (times != nullptr) DeviceSync();
    return std::chrono::steady_clock::now();
  };
  auto since = [&](std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(now() - t0).count();
  };

  // pass 1: the counts of every chunk
  auto t0 = now();
  DArray<uint64_t> packed(n_chunks);
  {
    uint64_t* out = packed.data();
    ForEach(n_chunks, [=] __device__(size_t c) {
      const int64_t row = static_cast<int64_t>(c) / per_row;
      const int64_t x0 = (static_cast<int64_t>(c) - row * per_row) * chunk;
      const int64_t x1 = x0 + chunk < wc ? x0 + chunk : wc;
      CountInChunk counter;
      counter.want_edges = want_edges && row < n_edge_rows;
      WalkChunk<T, Turn, CountInChunk>(raster, y0 + row, x0, x1, counter);
      out[c] = counter.packed;
    });
  }
  if (times != nullptr) times->count += since(t0);

  // offsets: within the block per chunk, of the block per list
  t0 = now();
  DArray<int64_t> start[5];
  int64_t total[5] = {0, 0, 0, 0, 0};
  for (int k = 0; k < 5; k++) start[k].Reset(n_blocks);
  {
    uint64_t* io = packed.data();
    int64_t* b0 = start[0].data();
    int64_t* b1 = start[1].data();
    int64_t* b2 = start[2].data();
    int64_t* b3 = start[3].data();
    int64_t* b4 = start[4].data();
    ForEach(n_blocks, [=] __device__(size_t b) {
      const int64_t first = static_cast<int64_t>(b) * per_block;
      const int64_t last = first + per_block < n_chunks ? first + per_block : n_chunks;
      uint64_t sum = 0;
      for (int64_t c = first; c < last; c++) {
        uint64_t counts = io[c];
        io[c] = sum | (counts != 0 ? kChunkWrites : 0);
        sum += counts;
      }
      b0[b] = static_cast<int64_t>(sum & kCountMask);
      b1[b] = static_cast<int64_t>((sum >> kCountBits) & kCountMask);
      b2[b] = static_cast<int64_t>((sum >> (2 * kCountBits)) & kCountMask);
      b3[b] = static_cast<int64_t>((sum >> (3 * kCountBits)) & kCountMask);
      b4[b] = static_cast<int64_t>((sum >> (4 * kCountBits)) & kCountMask);
    });
  }
  for (int k = 0; k < 5 && n_blocks > 0; k++) {
    int64_t last = start[k].Get(n_blocks - 1);
    thrust::exclusive_scan(Exec(), start[k].begin(), start[k].end(), start[k].begin(), int64_t{0});
    total[k] = start[k].Get(n_blocks - 1) + last;
  }
  for (int d = 0; d < 4; d++) {
    pos[d].Reset(total[d]);
    if (d == 1 || d == 3) flag[d].Reset(total[d]);
  }
  if (want_edges) edges->Reset(total[4]);
  if (times != nullptr) times->offsets += since(t0);

  // pass 2: write
  t0 = now();
  {
    WriteChunk blank;
    for (int d = 0; d < 4; d++) {
      blank.pos[d] = pos[d].data();
      blank.flag[d] = flag[d].data();
    }
    blank.edges = want_edges ? edges->data() : nullptr;
    blank.wc = wc;
    blank.hc = hc;
    const uint64_t* in = packed.data();
    const int64_t* b0 = start[0].data();
    const int64_t* b1 = start[1].data();
    const int64_t* b2 = start[2].data();
    const int64_t* b3 = start[3].data();
    const int64_t* b4 = start[4].data();
    ForEach(n_chunks, [=] __device__(size_t c) {
      const uint64_t at = in[c];
      if ((at & kChunkWrites) == 0) return;
      const int64_t row = static_cast<int64_t>(c) / per_row;
      const int64_t x0 = (static_cast<int64_t>(c) - row * per_row) * chunk;
      const int64_t x1 = x0 + chunk < wc ? x0 + chunk : wc;
      const int64_t b = static_cast<int64_t>(c) / per_block;
      WriteChunk writer = blank;
      writer.at[0] = b0[b] + static_cast<int64_t>(at & kCountMask);
      writer.at[1] = b1[b] + static_cast<int64_t>((at >> kCountBits) & kCountMask);
      writer.at[2] = b2[b] + static_cast<int64_t>((at >> (2 * kCountBits)) & kCountMask);
      writer.at[3] = b3[b] + static_cast<int64_t>((at >> (3 * kCountBits)) & kCountMask);
      writer.at[4] = b4[b] + static_cast<int64_t>((at >> (4 * kCountBits)) & kCountMask);
      writer.y = y0 + row;
      writer.want_edges = want_edges && row < n_edge_rows;
      WalkChunk<T, Turn, WriteChunk>(raster, y0 + row, x0, x1, writer);
    });
  }
  if (times != nullptr) times->write += since(t0);
}

// Step 1 on a strip as the configuration asks: chunk = 0 is s1_corners/base.h, chunk = -1 the corners of 32
// bytes of cells.
template <typename T, typename Turn = TurnByValue>
void FindVerticesInStripAs(int64_t chunk, const DRasterView<T>& raster, int64_t y0, int64_t n_rows,
                           int64_t n_edge_rows, int64_t wc, int64_t hc, DArray<int64_t> pos[4],
                           DArray<uint8_t> flag[4], DArray<int64_t>* edges, TwoPassTimes* times = nullptr) {
  if (chunk < 0) chunk = static_cast<int64_t>(32 / sizeof(T));
  if (chunk > 0) {
    FindVerticesInStripTwoPass<T, Turn>(raster, y0, n_rows, n_edge_rows, wc, hc, chunk, pos, flag, edges, times);
  } else {
    FindVerticesInStrip<T, Turn>(raster, y0, n_rows, n_edge_rows, wc, hc, pos, flag, edges);
  }
}

}  // namespace xrasp
#endif  // XRASP_STEPS_S1_CORNERS_OPT_TWO_PASS_H
