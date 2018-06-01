#include "predict.h"
#include "trajectory.h"
#include "../constants.h"
#include <vector>

using namespace std;

vector<Path::Trajectory> Path::Predict::calc(const vector<VehicleData::OtherVehicleData>  &otherVehicles, size_t numPrevious) const {
  vector<Path::Trajectory> predictions;
  for (auto otherVehicle = otherVehicles.begin(); otherVehicle != otherVehicles.end(); ++otherVehicle) {
    Path::Trajectory trajectory;
    if (otherVehicle->isFrenetValid && (otherVehicle->vs > 5.0)) {
      trajectory = inFrenetSpace(*otherVehicle);
    } else if (otherVehicle->speed > 0.0) {
      trajectory = inCartesianSpace(*otherVehicle);
    } else {
      trajectory = standStill(*otherVehicle);
    }
    /* Erase prediction matching trajectory already presented to the simulator. */
    trajectory.erase(0u, numPrevious - 1u);
    predictions.push_back(trajectory);
  }
  return predictions;
}

Path::Trajectory Path::Predict::inFrenetSpace(const VehicleData::OtherVehicleData &otherVehicle) const {
  /* Calculate the lane separated into an integral number and a fraction, where the later is used to understand
   * position within lane regardless of lane number. */
  const double lane = otherVehicle.d / constants.laneWidth;
  double laneIntegralPart;
  const double laneFraction = modf(lane, &laneIntegralPart);
  const int currentLane = static_cast<int>(laneIntegralPart);
  Logic::Intention intention;
  int nextLane;
  if (otherVehicle.vd > 1.0) {
    intention = Logic::Intention::LaneChangeRight;
    // Look at lane fraction value to decide target lane.
    nextLane = (laneFraction > 0.5) ? currentLane + 1 : currentLane;
  } else if (otherVehicle.vd < -1.0) {
    intention = Logic::Intention::LaneChangeLeft;
    // Look at lane fraction value to decide target lane.
    nextLane = (laneFraction < 0.5) ? currentLane - 1 : currentLane;
  } else {
    intention = Logic::Intention::KeepLane;
    nextLane = currentLane;
  }

  /* Calculate remaining change in d direction. */
  const double target_d = nextLane * constants.laneWidth + (constants.laneWidth / 2.0);
  const double delta_d = target_d - otherVehicle.d;

  /* Calculated the distance at which the lane change is completed by assuming constant velocity both in s and d
   * direction, although with some restriction on how long time the lane change may take. */
  const double maxLaneChangeTime = 3.0;
  const double minLaneAdjustmentTime = 0.5;
  const double delta_t = max(minLaneAdjustmentTime, min(maxLaneChangeTime, delta_d / otherVehicle.vd));
  const double delta_s = otherVehicle.vs * delta_t;
  const double delta_speed = 0.0;
  Path::Trajectory trajectory = Path::TrajectoryCalculator::AdjustSpeed(intention, nextLane, otherVehicle, delta_s, delta_d, delta_speed);

  if (trajectory.size() < minPredictionLength) {
    /* Extend prediction but now assume it continues in same lane. */
    trajectory += Path::TrajectoryCalculator::ConstantSpeed(intention, trajectory.getEndState(otherVehicle), minPredictionLength - trajectory.size());
  }
  return trajectory;
}

Path::Trajectory Path::Predict::inCartesianSpace(const VehicleData::OtherVehicleData &otherVehicle) const {
  Path::Trajectory trajectory = Path::TrajectoryCalculator::Others(otherVehicle, minPredictionLength);
  return trajectory;
}

Path::Trajectory Path::Predict::standStill(const VehicleData::OtherVehicleData &otherVehicle) const {
  Path::Trajectory trajectory;
  for (size_t i = 0u; i < minPredictionLength; ++i) {
    trajectory.push_back(otherVehicle.x, otherVehicle.y, Logic::Unknown, TARGET_LANE_UNKNOWN);
  }
  return trajectory;
}
