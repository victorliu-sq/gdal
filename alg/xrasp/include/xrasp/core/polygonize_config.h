#ifndef XRASP_CORE_POLYGONIZE_CONFIG_H
#define XRASP_CORE_POLYGONIZE_CONFIG_H
// What a run is told to do.  The switches below choose, per step, which file under steps/ does the work:
//
//   switch        step(s)                       file                                  off / on
//   ------        -------                       ----                                  --------
//   mode          how the raster reaches s1     core/polygonizer_strips.h             whole | strips
//                 vertices beyond GPU memory    core/polygonizer_bands.h, steps/bands  bands
//   corner_chunk  s1 corners                    s1_corners/opt_two_pass.h             0 = s1_corners/base.h
//   link_index    s2 links                      s2_links/opt_line_index.h             false = s2_links/base.h alone
//   tile          s3 rank                       s3_rank/opt_two_level.h               0 = s3_rank/base.h
//                 s4 diagonals                  s4_diagonals/opt_two_level.h          0 = s4_diagonals/base.h
//   labels        s0 label blocks               s0_labels/union_find.h                false = step skipped
//                 s1 corners                    s1_corners/with_labels.h              false = s1_corners/base.h
//                 s4 diagonals                  (nothing to repair)                   false = s4_diagonals/*
//                 s6 holes                      s6_holes/by_label.h                   false = s6_holes/by_west_search.h
#include <cstdint>
#include <string>

namespace xrasp {

struct PolygonizeConfig {
  // input and output
  std::string raster_path;
  std::string json_file;
  bool overwrite = false;
  std::string output_dir;
  std::string check_file;
  // how to run
  std::string mode = "whole";
  uint64_t strip_cells = 1ull << 27;
  uint64_t band_vertices = 100000000;      // mode=bands: vertices per band
  int repeat = 1;
  int warmup = 0;
  // optimizations
  int corner_chunk = 0;                    // corners per GPU thread in the two-pass s1 (0: s1_corners/base.h,
                                           // -1: as many as there are cells in 32 bytes)
  bool bitmap = false;                     // mode=whole: s1 and s2 on bitmaps (s1_corners/opt_bitmap.h)
  bool pieces = false;                     // with tile > 0: s3 to s7 per piece (core/piece_pipeline.h)
  bool walk_experiment = false;            // with bitmap and pieces: time the piece walk on links and on bitmaps
  bool keep_raster = true;                 // mode=whole: the raster stays on the device to the end (the value of
                                           // every ring is read there); false: it is freed after step 1 and the
                                           // values are looked up on the host
  int walk_share = 1;                      // with pieces: pieces per thread in the two walks
  bool link_index = false;                 // search the next vertex only in its own row or column
  int tile = 0;
  bool rank_walk = false;                  // with tile > 0: level 1 by walking every piece (s3_rank/opt_two_level.h)
  bool open_walk = false;                  // mode=bands: the outlines that leave a band found by walking them from
                                           // where they enter it (steps/bands/b1_open_vertices.h)
  bool band_on_device = false;             // mode=bands: everything in device memory -- the raster from the start, the
                                           // finished rings, the set-aside vertices, the result (its vertices one piece
                                           // per band); the host computes nothing
  bool labels = false;
  // GPU memory
  bool pool = true;
  uint64_t memory_limit_bytes = 0;
};

}  // namespace xrasp
#endif  // XRASP_CORE_POLYGONIZE_CONFIG_H
