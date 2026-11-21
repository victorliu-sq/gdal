#ifndef XRASP_UTIL_NPY_H
#define XRASP_UTIL_NPY_H
// Reading and writing NumPy .npy files (version 1.0 / 2.0, C order, little endian): how rasters come in
// (data/targets/scripts/to_npy.py converts a GeoTIFF once) and how output arrays go out.
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace xrasp {

struct NpyHeader {
  std::string descr;            // "|u1", "<i2", ...
  std::vector<int64_t> shape;
  size_t data_offset = 0;
};

inline NpyHeader ReadNpyHeader(FILE* f, const std::string& path) {
  unsigned char magic[10];
  if (fread(magic, 1, 8, f) != 8 || magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' ||
      magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
    throw std::runtime_error(path + ": not a .npy file");
  }
  size_t header_len = 0;
  if (magic[6] == 1) {
    unsigned char len[2];
    if (fread(len, 1, 2, f) != 2) throw std::runtime_error(path + ": truncated header");
    header_len = len[0] | (len[1] << 8);
  } else {
    unsigned char len[4];
    if (fread(len, 1, 4, f) != 4) throw std::runtime_error(path + ": truncated header");
    header_len = len[0] | (len[1] << 8) | (len[2] << 16) | (static_cast<size_t>(len[3]) << 24);
  }
  std::string text(header_len, ' ');
  if (fread(text.data(), 1, header_len, f) != header_len) throw std::runtime_error(path + ": truncated header");
  NpyHeader h;
  h.data_offset = static_cast<size_t>(ftell(f));
  auto value_of = [&](const std::string& key) {
    size_t at = text.find("'" + key + "'");
    if (at == std::string::npos) throw std::runtime_error(path + ": no '" + key + "' in the header");
    return text.find(':', at) + 1;
  };
  {
    size_t at = text.find('\'', value_of("descr"));
    h.descr = text.substr(at + 1, text.find('\'', at + 1) - at - 1);
  }
  if (text.find("True", value_of("fortran_order")) < text.find(',', value_of("fortran_order"))) {
    throw std::runtime_error(path + ": Fortran order is not supported");
  }
  {
    size_t open = text.find('(', value_of("shape")), close = text.find(')', open);
    std::string dims = text.substr(open + 1, close - open - 1);
    size_t pos = 0;
    while (pos < dims.size()) {
      size_t comma = dims.find(',', pos);
      std::string tok = dims.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
      if (tok.find_first_of("0123456789") != std::string::npos) h.shape.push_back(std::stoll(tok));
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
  }
  return h;
}

// descr of a C++ integer type
template <typename T>
inline std::string NpyDescr() {
  std::string kind = (static_cast<T>(-1) < static_cast<T>(0)) ? "i" : "u";
  return (sizeof(T) == 1 ? "|" : "<") + kind + std::to_string(sizeof(T));
}

template <typename T>
inline void WriteNpy(const std::string& path, const T* data, size_t n) {
  std::string dict = "{'descr': '" + NpyDescr<T>() + "', 'fortran_order': False, 'shape': (" +
                     std::to_string(n) + ",), }";
  size_t total = 10 + dict.size() + 1;
  dict.append((64 - total % 64) % 64, ' ');
  dict.push_back('\n');
  FILE* f = fopen(path.c_str(), "wb");
  if (f == nullptr) throw std::runtime_error("cannot write " + path);
  const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  unsigned char len[2] = {static_cast<unsigned char>(dict.size() & 0xff),
                          static_cast<unsigned char>(dict.size() >> 8)};
  fwrite(magic, 1, 8, f);
  fwrite(len, 1, 2, f);
  fwrite(dict.data(), 1, dict.size(), f);
  fwrite(data, sizeof(T), n, f);
  fclose(f);
}

}  // namespace xrasp
#endif  // XRASP_UTIL_NPY_H
