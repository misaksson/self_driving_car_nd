#ifndef LOGIC_H
#define LOGIC_H

#include <vector>

namespace Path {
  class Logic {
  public:
    Logic() {};
    virtual ~Logic() {};

    enum Intention {
      KeepLane = 0,
      LaneChangeLeft,
      LaneChangeRight,
      PrepareLaneChangeLeft,
      PrepareLaneChangeRight,
    };

    /** Get a list of intentions to evaluate for current state.  */
    std::vector<Intention> GetIntentionsToEvaluate(double d) const;
  };
};
#endif /* LOGIC_H */
