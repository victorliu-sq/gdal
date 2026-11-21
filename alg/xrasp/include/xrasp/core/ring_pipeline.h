#ifndef XRASP_CORE_RING_PIPELINE_H
#define XRASP_CORE_RING_PIPELINE_H
// Steps 2 to 5 on one set of vertex lists: next vertex, ring and position, diagonal repair, rings.
//
// The lists need not hold all vertices of the raster, as long as they are made of whole rings: the next
// vertex of a vertex is the nearest one ahead in the full list, and it is in the set, so it is also the
// nearest one in the set.  The strip version runs this once on everything; the band version runs it on the
// rings that lie inside a band, and once more on the rings that cross bands.
#include "xrasp/steps/s2_links/base.h"
#include "xrasp/steps/s3_rank/base.h"
#include "xrasp/steps/s3_rank/opt_two_level.h"
#include "xrasp/steps/s4_diagonals/base.h"
#include "xrasp/steps/s4_diagonals/opt_two_level.h"
#include "xrasp/steps/s5_rings/base.h"
#include "xrasp/structs/rings.h"
#include "xrasp/structs/vertex_lists.h"
#include "xrasp/util/dev_array.h"

namespace xrasp {

struct RingPipeline {
  DArray<int32_t> nxt, vx, vy;            // per vertex: next vertex, corner
  DArray<int32_t> m, off;                 // per vertex: smallest vertex of its ring, steps to it
  DArray<int32_t> pair_a, pair_b;         // the two vertices of one value at every diagonal corner
  TwoLevel two;                           // tile > 0: what the two-level ranking keeps from step 3 to step 4
  Rings rings;
  int64_t pairs = 0, one_block = 0;
  int rounds = 0, rounds_level2 = 0, rounds_repair = 0;
  int64_t pieces = 0;
  RankProfile prof;                       // where the time of step 3 went (two-level)

  // s2
  void Link(const VertexLists& lists) { LinkVertices(lists, &nxt, &vx, &vy); }
  // s2 where the next vertex of every vertex is known already: no search
  void UseLinks(DArray<int32_t>&& links, const VertexLists& lists) {
    nxt = std::move(links);
    CornersOfVertices(lists, &vx, &vy);
  }
  // s3.  tile = 0: all vertices together; otherwise in two levels with tiles of that size (walk: level 1 by
  // walking every piece instead of pointer jumping)
  // In two levels, giving back what of the lists the later steps do not read (the vertices entered moving
  // down stay): the pairs are found first, the ranking itself only needs the counts.
  void RankAndRelease(VertexLists* lists, int tile, bool walk) {
    FindDiagonalPairs(*lists, &pair_a, &pair_b);
    lists->FreeAllButDown();
    RankTwoLevel(*lists, nxt, vx, vy, pair_a, pair_b, tile, &m, &off, &two, walk);
    rounds = two.rounds1;
    rounds_level2 = two.rounds2;
    pieces = two.pieces;
      prof = two.prof;
  }
  void Rank(const VertexLists& lists, int tile, bool walk = false) {
    if (tile > 0) {
      FindDiagonalPairs(lists, &pair_a, &pair_b);     // the diagonal corners are cut points
      RankTwoLevel(lists, nxt, vx, vy, pair_a, pair_b, tile, &m, &off, &two, walk);
      rounds = two.rounds1;
      rounds_level2 = two.rounds2;
      pieces = two.pieces;
      prof = two.prof;
    } else {
      rounds = RankFlat(nxt, &m, &off);
    }
  }
  // s4
  void Repair(const VertexLists& lists, int tile) {
    DiagonalRepair r;
    if (tile > 0 && two.pieces > 0) {
      r = RepairDiagonalsTwoLevel(pair_a, pair_b, &nxt, &m, &off, &two);
    } else {
      if (tile <= 0) FindDiagonalPairs(lists, &pair_a, &pair_b);
      r = RepairDiagonalsFlat(pair_a, pair_b, &nxt, &m, &off);
    }
    pair_a.Free();
    pair_b.Free();
    pairs = r.pairs;
    one_block = r.one_block;
    rounds_repair = r.rounds;
  }
  // s5.  Afterwards only rings, vx and vy are left (and nxt, when the caller keeps the links).
  void Classify(const VertexLists& lists, bool keep_links = false) {
    BuildRings(lists, nxt, vx, vy, &m, &off, &rings);
    if (!keep_links) nxt.Free();
  }
  // counts: a json object (the programs), or anything else that takes counts["name"] = number
  template <typename Counts>
  void AddCounts(Counts* counts) const {
    (*counts)["JumpRounds"] = rounds;
    if (pieces > 0) {
      (*counts)["Pieces"] = pieces;
      (*counts)["Level2Rounds"] = rounds_level2;
      if (two.walk_cuts > 0) (*counts)["WalkCuts"] = two.walk_cuts;
      (*counts)["RankCutsTime"] = prof.cuts_ms;
      (*counts)["RankLevel1Time"] = prof.level1_ms;
      (*counts)["RankNodesTime"] = prof.nodes_ms;
      (*counts)["RankLevel2Time"] = prof.level2_ms;
      if (prof.tiles_used > 0) {
        (*counts)["TilesUsed"] = prof.tiles_used;
        (*counts)["TileVerticesMax"] = prof.tile_max;
        (*counts)["TileVerticesP50"] = prof.tile_p50;
        (*counts)["TileVerticesP99"] = prof.tile_p99;
        (*counts)["TilesOver6000"] = prof.tiles_over_6k;
        (*counts)["VerticesInTilesOver6000"] = prof.vertices_over_6k;
        (*counts)["TilesOver12000"] = prof.tiles_over_12k;
        (*counts)["VerticesInTilesOver12000"] = prof.vertices_over_12k;
      }
    }
    (*counts)["DiagonalPairs"] = pairs;
    (*counts)["DiagonalPairsOneBlock"] = one_block;
    (*counts)["JumpRounds2"] = rounds_repair;
    (*counts)["Rings"] = rings.nr;
    (*counts)["Holes"] = rings.nholes;
  }
};

}  // namespace xrasp
#endif  // XRASP_CORE_RING_PIPELINE_H
