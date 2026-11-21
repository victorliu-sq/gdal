#ifndef XRASP_UTIL_JSON_H
#define XRASP_UTIL_JSON_H
// The stand-in of GDAL for xrasp/util/json.h of the X-Rasp library (which is nlohmann/json there): the band
// version keeps the counts of a run in a Json (core/polygonizer_bands.h), and nothing inside GDAL reads them.
// This file is the fork's own, not a copy: measure/common/targets/scripts/sync_gdal leaves it alone.
namespace xrasp {

class Json {
 public:
  struct Slot {
    template <typename T>
    const Slot& operator=(const T&) const { return *this; }
  };
  static Json object() { return Json(); }
  Slot operator[](const char*) const { return Slot(); }
};

}  // namespace xrasp
#endif  // XRASP_UTIL_JSON_H
