#ifndef XRASP_UTIL_PARTS_H
#define XRASP_UTIL_PARTS_H
// Where the time of a run goes, finer than its steps.  With XRASP_PARTS in the environment a PartClock adds
// the milliseconds between two of its calls to a named part, and the program writes the parts with the counts
// ("Part <name>").  Without it nothing is timed (on the GPU a part ends with a wait for the device).
//
//   PartClock part;
//   A();  part("a");        // A's time -> "a"
//   B();  part("b");
#include <chrono>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "xrasp/util/device_memory.h"

namespace xrasp {

struct PartsLog {
  bool on = std::getenv("XRASP_PARTS") != nullptr;
  std::vector<std::pair<std::string, double>> parts;      // in the order they first ran

  void Add(const std::string& name, double ms) {
    if (!on) return;
    for (auto& p : parts) {
      if (p.first == name) {
        p.second += ms;
        return;
      }
    }
    parts.emplace_back(name, ms);
  }
  static PartsLog& get() {
    static PartsLog log;
    return log;
  }
};

class PartClock {
 public:
  PartClock() : on_(PartsLog::get().on) { Reset(); }
  // the clock starts again; what ran since the last call is given to no part
  void Reset() {
    if (!on_) return;
    DeviceSync();
    t_ = std::chrono::steady_clock::now();
  }
  // milliseconds since the last call, given to no part
  double Lap() {
    if (!on_) return 0;
    DeviceSync();
    auto now = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - t_).count();
    t_ = now;
    return ms;
  }
  void operator()(const std::string& name) {
    if (!on_) return;
    PartsLog::get().Add(name, Lap());
  }

 private:
  bool on_;
  std::chrono::steady_clock::time_point t_;
};

}  // namespace xrasp
#endif  // XRASP_UTIL_PARTS_H
