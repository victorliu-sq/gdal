#ifndef XRASP_STRUCTS_DEV_RASTER_H
#define XRASP_STRUCTS_DEV_RASTER_H
// A strip of the raster on the GPU: cell rows row0 .. row0 + rows - 1 of an h x w raster.  At(x, y) is the
// value of cell (x, y), or kOutside when the cell is outside the raster.  (Cells inside the raster but
// outside the strip must not be asked for.)
#include <cstdint>

#include "xrasp/backend/macros.h"

namespace xrasp {

constexpr int64_t kOutside = INT64_MIN;

template <typename T>
struct DRasterView {
  const T* data;
  int64_t row0, rows;
  int64_t h, w;

  DEV_INLINE int64_t At(int64_t x, int64_t y) const {
    if (x < 0 || x >= w || y < 0 || y >= h) return kOutside;
    return static_cast<int64_t>(data[(y - row0) * w + x]);
  }
};

}  // namespace xrasp
#endif  // XRASP_STRUCTS_DEV_RASTER_H
