#include "logic.h"
#include "../constants.h"
#include "../helpers.h"
#include <iostream>
#include <math.h>
#include <vector>
#include <tuple>

using namespace std;

vector<Path::Logic::Intention> Path::Logic::GetIntentionsToEvaluate(double d) const {
  int currentLane = Helpers::GetLane(d);
  vector<Intention> intensions;
  intensions.push_back(KeepLane);
  if (currentLane > 0) {
    intensions.push_back(LaneChangeLeft);
    intensions.push_back(PrepareLaneChangeLeft);
    if (currentLane > 1) {
      intensions.push_back(TwoLaneChangesLeft);
    }
  }
  if (currentLane < (constants.numLanes - 1)) {
    intensions.push_back(LaneChangeRight);
    intensions.push_back(PrepareLaneChangeRight);
    if (currentLane < (constants.numLanes - 2)) {
      intensions.push_back(TwoLaneChangesRight);
    }
  }
  return intensions;
}
