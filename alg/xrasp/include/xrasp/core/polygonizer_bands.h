#ifndef XRASP_CORE_POLYGONIZER_BANDS_H
#define XRASP_CORE_POLYGONIZER_BANDS_H
// mode=bands: neither the raster nor its vertices have to fit the GPU.
//
// The strips of the strip version are grouped into bands of about `band_vertices` vertices.  In a band, the
// rings that lie inside it are finished on the GPU and moved to host memory; the vertices of outlines that
// leave the band (a few per cent) are set aside.  At the end those vertices, from all bands, go through the
// same steps once.  Holes whose block's other ring is among the set-aside vertices are attached then.
//
// core/run_polygonize.cu calls the member functions below in order, band after band.
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "xrasp/core/polygonize_config.h"
#include "xrasp/core/ring_pipeline.h"
#include "xrasp/structs/host_polygons.h"
#include "xrasp/structs/raster.h"
#include "xrasp/structs/vertex_lists.h"
#include "xrasp/util/dev_array.h"
#include "xrasp/util/json.h"

namespace xrasp {

// What a band hands to a sink (PolygonizerBands::SetBandSink) when its rings are finished: the rings of the band
// one after the other, and the same rings group after group.  A group whose first ring is not a hole is a
// polygon that is complete -- its outer ring first, then its holes, all of which lie in this band; a group
// whose first ring is a hole holds holes of a polygon whose outline crosses bands, and is not a polygon yet.
// The pointers are to device memory (host memory on the CPU) and hold only while the sink runs.
struct BandPolygons {
  int64_t rings = 0, groups = 0;
  const int32_t* x = nullptr;               // the vertices, ring after ring
  const int32_t* y = nullptr;
  const int64_t* ring_offsets = nullptr;    // rings + 1: where the vertices of every ring start
  const int64_t* order = nullptr;           // rings: the rings group after group
  const int64_t* group_start = nullptr;     // groups + 1: where every group starts in order
  const uint8_t* is_hole = nullptr;         // rings
  const int64_t* value = nullptr;           // rings: the value of the ring's block
};

class PolygonizerBands {
 public:
  PolygonizerBands(const PolygonizeConfig& config, const Raster& raster);

  bool HasStrips() const { return next_row_ < hc_; }
  void FindVerticesOfNextStrip();   // s1 on the next strip; its vertices join the current band
  bool BandIsComplete() const;      // enough vertices for a band, or no strip left
  void SortBand();                  // the band's vertex lists
  void SplitBand();                 // b1: open vertices are set aside, the others stay
  void CloseBandRings();            // s2 .. s5 on the rings inside the band, b2: their holes
  void BandToHost();                // the finished rings leave the GPU
  void CloseOpenRings();            // s2 .. s5 on the set-aside vertices of all bands, their holes
  void MergeRings();                // waiting holes, polygons, the output arrays
  void ResultToHost();              // band_on_device: the result copied to host memory, for polygons()

  // band_on_device: a host application takes the complete polygons of every band as soon as the band is done
  // (to write them out), and they are given back: what stays to the end, and what pieces() and result_*() hold
  // then, is only the polygons whose outline crosses bands.  Set before the first strip.
  //
  // at_end = false: the sink is called when a band is done, between two bands.
  // at_end = true: nothing is handed out while the raster is polygonized; the bands stay as they are, and
  // DrainBands() hands them out one by one afterwards, each given back as soon as the sink returns.  The
  // caller takes the polygons that cross bands (pieces(), result_*()) BEFORE it drains: their holes are
  // rings of the bands.
  void SetBandSink(std::function<void(const BandPolygons&)> sink, bool at_end = false) {
    sink_ = std::move(sink);
    sink_at_end_ = at_end;
  }
  void DrainBands();

  const HostPolygons& polygons() const { return out_; }
  // band_on_device: the result where MergeRings() left it, for a host application that writes it out itself.
  // The tables are those of structs/host_polygons.h with ring_index; the vertices of ring r are in the piece
  // that holds it, pieces()[k] holding rings first_ring .. first_ring + rings - 1 one after the other.
  // ring_starts: null when the vertices of the piece's rings follow each other from the start of x (ring r
  // then starts at ring_offsets[r] - ring_offsets[first_ring]); otherwise where each of the piece's rings
  // starts in x (a band that still holds the polygons it has to hand out: DrainBands()).
  struct Piece {
    const int32_t* x;
    const int32_t* y;
    int64_t vertices, rings, first_ring, first_vertex;
    const int64_t* ring_starts;
  };
  std::vector<Piece> pieces() const {
    std::vector<Piece> all;
    int64_t ring = 0, vertex = 0;
    for (const auto& c : chunks_) {
      all.push_back({c.dx.data(), c.dy.data(), static_cast<int64_t>(c.dx.size()), c.n_rings, ring, vertex,
                     c.d_start.empty() ? nullptr : c.d_start.data()});
      ring += c.n_rings;
      vertex += static_cast<int64_t>(c.dx.size());
    }
    return all;
  }
  const DArray<int64_t>& result_values() const { return r_values_; }
  const DArray<int64_t>& result_poly_offsets() const { return r_poly_offsets_; }
  const DArray<int64_t>& result_ring_index() const { return r_ring_index_; }
  const DArray<int64_t>& result_ring_offsets() const { return r_ring_offsets_; }
  const Json& counts() const { return counts_; }

 private:
  // what a band (or the final pass) leaves in host memory
  struct Chunk {
    HostArray<int32_t> x, y;                    // vertices in ring order
    std::vector<int32_t> ring_len;
    std::vector<uint8_t> is_hole;
    std::vector<int64_t> parent, pending, value;
    // band_on_device: the same in device memory, and nothing above
    DArray<int32_t> dx, dy, d_len;
    DArray<uint8_t> d_hole;
    DArray<int64_t> d_parent, d_pending, d_value;
    int64_t n_rings = 0;
    // a band that waits to be drained (sink at the end): dx, dy are all its rings; d_start says where the
    // n_rings rings that stay start in them; and its rings as the sink gets them
    DArray<int64_t> d_start;
    DArray<int64_t> a_offsets, a_order, a_groups, a_value;
    DArray<uint8_t> a_hole;
    int64_t n_all = 0, n_groups = 0;
  };
  // the down-entered vertices of the rings finished in a band: sorted positions and their (global) rings
  struct BandIndex {
    int64_t y_first;
    std::vector<int64_t> pos, ring;
    DArray<int64_t> d_pos, d_ring;              // band_on_device
  };
  template <typename T>
  void FindVerticesOfNextStripOfType();
  void RingsToChunk(Chunk* chunk);
  void MergeRingsOnDevice();
  void EmitBand(Chunk* chunk, BandIndex* index);

  const PolygonizeConfig& config_;
  const Raster& raster_;
  int64_t h_, w_, wc_, hc_;
  int64_t rows_;                                // corner rows per strip
  int64_t next_row_ = 0;                        // first corner row of the next strip

  // the band being collected
  std::vector<DArray<int64_t>> pos_parts_[4], edge_parts_;
  std::vector<DArray<uint8_t>> flag_parts_[4];
  int64_t in_band_ = 0, band_first_row_ = 0;
  // the band being processed
  VertexLists lists_;                           // SortBand: all vertices; after SplitBand: the kept ones
  DArray<int64_t> edges_, pos0_all_;
  DArray<int32_t> owner_;
  DArray<int32_t> band_links_;                  // open_walk: the links of the kept vertices, from SplitBand
  RingPipeline pipe_;
  DArray<int64_t> parent_, pending_;
  bool band_has_rings_ = false;

  // host side
  std::vector<Chunk> chunks_;
  std::vector<BandIndex> band_index_;
  std::vector<DArray<int64_t>> open_dev_[4];    // band_on_device: the same in device memory
  std::vector<std::vector<int64_t>> open_[4];   // per direction: the packed set-aside vertices of every band
  std::vector<int64_t> open_pos0_, open_ring0_; // the set-aside down-entered vertices and their rings
  int64_t ring_base_ = 0;
  int64_t n_bands_ = 0, n_strips_ = 0, n_vertices_ = 0, n_open_ = 0, n_one_block_ = 0, n_emitted_rings_ = 0;
  double upload_ms_ = 0, tables_ms_ = 0, split_open_ms_ = 0, split_groups_ms_ = 0, split_links_ms_ = 0;

  // band_on_device: the raster, the set-aside down-entered vertices with their rings, the tables of the result
  StripCells<uint8_t> device_cells_;
  DArray<int64_t> d_open_pos0_, d_open_ring0_;
  DArray<int64_t> r_values_, r_poly_offsets_, r_ring_index_, r_ring_offsets_;
  bool result_on_host_ = false;
  std::function<void(const BandPolygons&)> sink_;
  bool sink_at_end_ = false;

  HostPolygons out_;
  Json counts_;
};

}  // namespace xrasp
#endif  // XRASP_CORE_POLYGONIZER_BANDS_H
