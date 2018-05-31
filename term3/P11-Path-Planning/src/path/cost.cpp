#include "cost.h"
#include "trajectory.h"
#include "../constants.h"
#include "../helpers.h"
#include "../vehicle_data.h"
#include <algorithm>
#include <assert.h>
#include <vector>

using namespace std;

/* Definition of static members. */
bool Path::Cost::verbose;
VehicleData Path::Cost::vehicleData;
Path::Trajectory Path::Cost::previousTrajectory;
std::vector<Path::Trajectory> Path::Cost::predictions;
VehicleData::EgoVehicleData Path::Cost::previousEgoEndState;
std::vector<VehicleData::EgoVehicleData> Path::Cost::othersEndState;
std::vector<std::vector<VehicleData::EgoVehicleData>> Path::Cost::othersStateSamples;
std::vector<Path::Trajectory::Kinematics> Path::Cost::othersKinematics;
VehicleData::EgoVehicleData Path::Cost::egoEndState;
std::vector<VehicleData::EgoVehicleData> Path::Cost::egoStateSamples;
Path::Trajectory::Kinematics Path::Cost::egoKinematics;
int Path::Cost::startLane;
int Path::Cost::endLane;
double Path::Cost::shortestDistanceToOthers;
double Path::Cost::shortestDistanceToOthersAhead;

/** Custom compare for abs max in a vector. */
static bool fabsCompare(double a, double b);

void Path::Cost::preprocessCommonData(const Path::Trajectory previousTrajectory_, const VehicleData vehicleData_, const std::vector<Path::Trajectory> predictions_) {
  /* Copy some data for convenience. */
  Path::Cost::previousTrajectory = previousTrajectory_;
  Path::Cost::vehicleData = vehicleData_;
  Path::Cost::predictions = predictions_;

  Path::Cost::previousEgoEndState = previousTrajectory.getEndState(vehicleData.ego);
  Path::Cost::othersKinematics.clear();
  Path::Cost::othersStateSamples.clear();
  for (size_t vehicleIdx = 0u; vehicleIdx < vehicleData.others.size(); ++vehicleIdx) {
    /* Calculate kinematics data for predicted trajectory of other vehicle. */
    Path::Cost::othersKinematics.push_back(predictions[vehicleIdx].getKinematics());

    /* Sample other vehicles state every 10th time-step along the predicted trajectory. */
    std::vector<VehicleData::EgoVehicleData> stateSamples;
    for (size_t trajectoryIdx = 1u; trajectoryIdx < predictions[vehicleIdx].size(); trajectoryIdx += 10u) {
      stateSamples.push_back(predictions[vehicleIdx].getState(vehicleData.others[vehicleIdx], trajectoryIdx));
    }
    Path::Cost::othersStateSamples.push_back(stateSamples);
  }
}

void Path::Cost::preprocessCurrentTrajectory(const Path::Trajectory &trajectory) {
  Path::Cost::egoEndState = trajectory.getEndState(previousEgoEndState);
  if (verbose) cout << "ego: " << vehicleData.ego << endl;
  if (verbose) cout << "end: " << egoEndState << endl;
  Path::Cost::egoKinematics = (previousTrajectory + trajectory).getKinematics();
  Path::Cost::othersEndState.clear();
  for (size_t i = 0u; i < vehicleData.others.size(); ++i) {
    assert(predictions[i].size() >= trajectory.size());
    Path::Cost::othersEndState.push_back(predictions[i].getState(vehicleData.others[i], trajectory.size() - 1));
    if (verbose) cout << vehicleData.others[i] << endl;
    if (verbose) cout << "end: " << Path::Cost::othersEndState.back() << endl;
  }

  /* Sample ego vehicle state every 10th time-step along the trajectory. */
  egoStateSamples.clear();
  for (size_t trajectoryIdx = 1u; trajectoryIdx < trajectory.size(); trajectoryIdx += 10u) {
    egoStateSamples.push_back(trajectory.getState(previousEgoEndState, trajectoryIdx));
  }

  startLane = Helpers::GetLane(vehicleData.ego.d);
  endLane = trajectory.size() < 2 ? startLane :
                                    Helpers::GetLane(trajectory.x.end()[-1],
                                                     trajectory.y.end()[-1],
                                                     Helpers::CalcYaw(trajectory.x.end()[-2],
                                                                      trajectory.y.end()[-2],
                                                                      trajectory.x.end()[-1],
                                                                      trajectory.y.end()[-1]));
  shortestDistanceToOthers = HUGE_VAL;
  for (size_t vehicleIdx = 0u; vehicleIdx < vehicleData.others.size(); ++vehicleIdx) {
    const double longitudinalDiff = Helpers::calcLongitudinalDiff(vehicleData.ego.s, vehicleData.others[vehicleIdx].s);
    if (longitudinalDiff > -8.0) {
      /* Skip vehicles behind ego vehicle. The constant speed predictions often suggest that vehicles approaching from
       * behind will collide with ego vehicle, effectively forcing the algorithm to change lane. In reality, the
       * vehicles coming from behind will slow down or change lane as they approach a slower vehicle. There are other
       * cost functions not utilizing this distance value to deal with scenarios where ego vehicle change lane in front
       * of other vehicles.
       */
      continue;
    }
    assert(predictions[vehicleIdx].size() >= trajectory.size());
    for (int i = 0; i < trajectory.size(); ++i) {
      shortestDistanceToOthers = min(shortestDistanceToOthers, Helpers::distance(trajectory.x[i], trajectory.y[i], predictions[vehicleIdx].x[i], predictions[vehicleIdx].y[i]));
    }
  }
  if (verbose) cout << "shortestDistanceToOthers = " << shortestDistanceToOthers << endl;

  shortestDistanceToOthersAhead = HUGE_VAL;
  for (size_t sampleIdx = 0u; sampleIdx < egoStateSamples.size(); ++sampleIdx) {
    int egoLane = Helpers::GetLane(egoStateSamples[sampleIdx].d);
    for (size_t vehicleIdx = 0u; vehicleIdx < vehicleData.others.size(); ++vehicleIdx) {
      /* Only consider vehicles in same lane that are ahead of ego vehicle at start. */
      if (Helpers::GetLane(othersStateSamples[vehicleIdx][sampleIdx].d) == egoLane &&
          Helpers::calcLongitudinalDiff(vehicleData.ego.s, vehicleData.others[vehicleIdx].s) < 0.0) {
        double distanceAhead = Helpers::calcLongitudinalDiff(othersStateSamples[vehicleIdx][sampleIdx].s, egoStateSamples[sampleIdx].s);
        if (distanceAhead >= -7.5) {
          shortestDistanceToOthersAhead = min(shortestDistanceToOthersAhead, distanceAhead);
        }
      }
    }
  }
  if (verbose) cout << "shortestDistanceToOthersAhead = " << shortestDistanceToOthersAhead << endl;
}

double Path::SlowSpeed::calc(const Path::Trajectory &trajectory) const {
  // Calculate cost for not reaching speed limit.
  const double cost = slowSpeedCostFactor * fabs(constants.speedLimit - egoKinematics.speeds.back());
  return cost;
}

double Path::ExceedSpeedLimit::calc(const Path::Trajectory &trajectory) const {
  // Calculate cost for exceeding speed limit.
  const double cost = (egoKinematics.speeds.back() > constants.speedLimit) ? exceedSpeedLimitCost : 0.0;
  return cost;
}

double Path::ChangeIntention::calc(const Path::Trajectory &trajectory) const {
  double cost;
  // Calculate cost for not keeping to previously intended decision.
  if ((previousEgoEndState.targetLane != trajectory.targetLane[0]) &&
      (previousEgoEndState.intention != Logic::Intention::None) &&
      (previousEgoEndState.intention != Path::Logic::Unknown)) {
    if (verbose) cout << "Changing intention from " << previousEgoEndState.intention <<
                         " with targetLane " << previousEgoEndState.targetLane <<
                         " to " << trajectory.intention[0] <<
                         " with targetLane " << trajectory.targetLane[0] << endl;
    cost = changeIntentionCost;
  } else {
    if (verbose) cout << "Sticking to intention " << previousEgoEndState.intention << " (" << trajectory.intention[0] << "?)" <<
                         " with targetLane " << trajectory.targetLane[0] << endl;
    cost = 0.0;
  }
  return cost;
}

double Path::LaneChange::calc(const Path::Trajectory &trajectory) const {
  // Calculate cost for lane changes.
  const double cost = laneChangeCostFactor * static_cast<double>(abs(endLane - startLane));
  return cost;
}

double Path::LaneChangeInFrontOfOther::calc(const Path::Trajectory &trajectory) const {
  double cost = 0.0;
  if (startLane != endLane) {
    for (auto other = vehicleData.others.begin(); other != vehicleData.others.end(); ++other) {
      if (Helpers::GetLane(other->d) == endLane) {
        const double longitudinalDiff = Helpers::calcLongitudinalDiff(vehicleData.ego.s, other->s);
        const double longitudinalTimeDiff = (longitudinalDiff > -15.0 && other->speed > 0.0) ? longitudinalDiff / other->speed : HUGE_VAL;
        if (longitudinalTimeDiff < 0.25) {
          cost = laneChangeInFrontOfOtherCost;
          break;
        }
      }
    }
  }
  return cost;
}

double Path::LaneChangeInFrontOfOtherFaster::calc(const Path::Trajectory &trajectory) const {
  double cost = 0.0;
  if (startLane != endLane) {
    for (auto other = vehicleData.others.begin(); other != vehicleData.others.end(); ++other) {
      if (Helpers::GetLane(other->d) == endLane) {
        const double longitudinalDiff = Helpers::calcLongitudinalDiff(vehicleData.ego.s, other->s);
        const double interceptTime = (longitudinalDiff > 0.0 && (other->speed - vehicleData.ego.speed) > 0.0) ? longitudinalDiff / (other->speed - vehicleData.ego.speed) : HUGE_VAL;
        if (interceptTime < 3.0) {
          cost = laneChangeInFrontOfOtherFasterCost;
          break;
        }
      }
    }
  }
  return cost;
}

double Path::NearOtherVehicles::calc(const Path::Trajectory &trajectory) const {
  // Calculate cost for driving near other vehicles.
  const double cost = inverseDistanceCostFactor / shortestDistanceToOthers;
  return cost;
}

double Path::Collision::calc(const Path::Trajectory &trajectory) const {
  double cost = 0.0;
  if (shortestDistanceToOthers < collisionDistance) {
    cost = collisionCost;
  }
  return cost;
}

double Path::SlowLane::calc(const Path::Trajectory &trajectory) const {
  // Calculate cost for going in a lane with slow vehicles ahead.
  double slowestSpeedAhead = constants.speedLimit;
  for (int i = 0; i < vehicleData.others.size(); ++i) {
    const double longitudinalDiff = Helpers::calcLongitudinalDiff(othersEndState[i].s, egoEndState.s);
    if ((Helpers::GetLane(othersEndState[i].d) == endLane) && (longitudinalDiff > 0.0)) {
      slowestSpeedAhead = min(slowestSpeedAhead, othersEndState[i].speed * (0.8 + longitudinalDiff / 200.0));
    }
  }
  const double cost = (constants.speedLimit - slowestSpeedAhead) * slowLaneCostFactor;
  return cost;
}

double Path::ViolateCriticalDistanceAhead::calc(const Path::Trajectory &trajectory) const {
  double cost = 0.0;
  const double longitudinalTimeDiff = shortestDistanceToOthersAhead / egoEndState.speed;
  if (longitudinalTimeDiff < criticalLongitudinalTimeDiff) {
    if (verbose) cout <<"Violated critical distance to vehicle ahead" << endl;
    cost = violateCriticalLongitudinalTimeDiffCost;
  }
  return cost;
}

double Path::Acceleration::calc(const Path::Trajectory &trajectory) const {
  auto maxAcceleration = max_element(egoKinematics.accelerations.begin(), egoKinematics.accelerations.end(), fabsCompare);
  const double cost = fabs(*maxAcceleration) * accelerationCostFactor;
  return cost;
}

double Path::Jerk::calc(const Path::Trajectory &trajectory) const {
  auto maxJerk = max_element(egoKinematics.jerks.begin(), egoKinematics.jerks.end(), fabsCompare);
  const double cost = fabs(*maxJerk) * jerkCostFactor;
  return cost;
}


double Path::YawRate::calc(const Path::Trajectory &trajectory) const {
  auto maxYawRate = max_element(egoKinematics.yawRates.begin(), egoKinematics.yawRates.end(), fabsCompare);
  const double cost = fabs(*maxYawRate) * yawRateCostFactor;
  return cost;
}

double Path::ExceedAccelerationLimit::calc(const Path::Trajectory &trajectory) const {
  auto maxAcceleration = max_element(egoKinematics.accelerations.begin(), egoKinematics.accelerations.end(), fabsCompare);
  const double cost = fabs(*maxAcceleration) > constants.accelerationLimit ? exceedAccelerationLimitCost : 0.0;
  return cost;
}

static bool fabsCompare(double a, double b) {
    return (fabs(a) < fabs(b));
}
