#ifndef XRASP_STRUCTS_RASTER_H
#define XRASP_STRUCTS_RASTER_H
// The input raster in host memory: h rows of w integer cells, in the type of the file.
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

#include <sys/mman.h>

#include "xrasp/util/npy.h"

namespace xrasp {

enum class CellType { kU8, kI8, kU16, kI16, kU32, kI32 };

inline const char* CellTypeName(CellType t) {
  switch (t) {
    case CellType::kU8: return "uint8";
    case CellType::kI8: return "int8";
    case CellType::kU16: return "uint16";
    case CellType::kI16: return "int16";
    case CellType::kU32: return "uint32";
    case CellType::kI32: return "int32";
  }
  return "?";
}

class Raster {
 public:
  int64_t h = 0, w = 0;
  CellType type = CellType::kU8;
  size_t cell_size = 1;

  template <typename T>
  const T* cells() const { return reinterpret_cast<const T*>(mapped_ != nullptr ? mapped_ : bytes_.get()); }
  uint8_t* bytes() { return bytes_.get(); }
  size_t num_cells() const { return static_cast<size_t>(h) * static_cast<size_t>(w); }
  size_t num_bytes() const { return num_cells() * cell_size; }
  void Allocate() { bytes_.reset(new uint8_t[num_bytes()]); }
  // The cells are `num_bytes()` bytes at `offset` of a file mapped read-only at `map` (`map_bytes` long).
  void UseMapping(void* map, size_t map_bytes, size_t offset) {
    map_ = std::shared_ptr<void>(map, [map_bytes](void* m) { munmap(m, map_bytes); });
    mapped_ = static_cast<const uint8_t*>(map) + offset;
  }

  // The cells are `num_bytes()` bytes of the caller's, which stay where they are while this raster is used
  // (a host application: the band of a MEM dataset inside GDAL).
  void UseCells(const void* cells) { mapped_ = static_cast<const uint8_t*>(cells); }

  // The value of cell (x, y), read on the host.
  int64_t Value(int64_t x, int64_t y) const {
    size_t i = static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x);
    switch (type) {
      case CellType::kU8: return cells<uint8_t>()[i];
      case CellType::kI8: return cells<int8_t>()[i];
      case CellType::kU16: return cells<uint16_t>()[i];
      case CellType::kI16: return cells<int16_t>()[i];
      case CellType::kU32: return cells<uint32_t>()[i];
      case CellType::kI32: return cells<int32_t>()[i];
    }
    return 0;
  }

 private:
  std::unique_ptr<uint8_t[]> bytes_;       // cells this raster owns (Allocate), or
  std::shared_ptr<void> map_;              // the file they are mapped from (UseMapping)
  const uint8_t* mapped_ = nullptr;
};

// A 2-D integer .npy file, mapped read-only and read in completely.  The cells are pages of the file, not
// memory of the program: when host memory runs short (rings moved to the host in band mode) the kernel
// drops them and reads them again from the file.
inline Raster LoadRaster(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (f == nullptr) throw std::runtime_error("cannot open " + path);
  NpyHeader header = ReadNpyHeader(f, path);
  if (header.shape.size() != 2) {
    fclose(f);
    throw std::runtime_error(path + ": expected a 2-D array");
  }
  Raster r;
  r.h = header.shape[0];
  r.w = header.shape[1];
  const std::string& d = header.descr;
  if (d == "|u1") { r.type = CellType::kU8; r.cell_size = 1; }
  else if (d == "|i1") { r.type = CellType::kI8; r.cell_size = 1; }
  else if (d == "<u2") { r.type = CellType::kU16; r.cell_size = 2; }
  else if (d == "<i2") { r.type = CellType::kI16; r.cell_size = 2; }
  else if (d == "<u4") { r.type = CellType::kU32; r.cell_size = 4; }
  else if (d == "<i4") { r.type = CellType::kI32; r.cell_size = 4; }
  else {
    fclose(f);
    throw std::runtime_error(path + ": cell type " + d + " is not supported");
  }
  long offset = ftell(f);
  fseek(f, 0, SEEK_END);
  long file_bytes = ftell(f);
  if (offset < 0 || file_bytes < 0 || static_cast<size_t>(file_bytes - offset) < r.num_bytes()) {
    fclose(f);
    throw std::runtime_error(path + ": file is shorter than its header says");
  }
  size_t map_bytes = static_cast<size_t>(offset) + r.num_bytes();
  void* map = mmap(nullptr, map_bytes, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fileno(f), 0);
  fclose(f);
  if (map == MAP_FAILED) throw std::runtime_error(path + ": cannot map the file");
  r.UseMapping(map, map_bytes, static_cast<size_t>(offset));
  return r;
}

}  // namespace xrasp
#endif  // XRASP_STRUCTS_RASTER_H
