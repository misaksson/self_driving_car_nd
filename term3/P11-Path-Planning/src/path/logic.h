#ifndef LOGIC_H
#define LOGIC_H

#include <ostream>
#include <vector>

namespace Path {
  class Logic {
  public:
    Logic() {};
    virtual ~Logic() {};

    enum Intention {
      None = -2,
      Unknown = -1,
      KeepLane = 0,
      LaneChangeLeft,
      LaneChangeRight,
      PrepareLaneChangeLeft,
      PrepareLaneChangeRight,
    };

    friend std::ostream& operator<<(std::ostream& os, const Intention value) {
      switch(value) {
        case None:
          os << "None";
          break;
        case Unknown:
          os << "Unknown";
          break;
        case KeepLane:
          os << "KeepLane";
          break;
        case LaneChangeLeft:
          os << "LaneChangeLeft";
          break;
        case LaneChangeRight:
          os << "LaneChangeRight";
          break;
        case PrepareLaneChangeLeft:
          os << "PrepareLaneChangeLeft";
          break;
        case PrepareLaneChangeRight:
          os << "PrepareLaneChangeRight";
          break;
      }
      return os;
    };

    /** Get a list of intentions to evaluate for current state.  */
    std::vector<Intention> GetIntentionsToEvaluate(double d) const;
  };
};
#endif /* LOGIC_H */
