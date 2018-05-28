#include "../constants.h"
#include "../helpers.h"
#include "../spline.h"
#include "logic.h"
#include "planner.h"
#include "trajectory.h"
#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace std;

/** Custom compare for abs max in vector. */
static bool fabsCompare(double a, double b);
/** Generate trajectories to evaluate for given state. */
static vector<Path::Trajectory> generateTrajectories(const VehicleData::EgoVehicleData &ego, vector<Path::Logic::Intention> intentionsToEvaluate);

Path::Planner::Planner(int minTrajectoryLength, int maxTrajectoryLength) :
    logic(Path::Logic()), predict(Path::Predict()),
    minTrajectoryLength(minTrajectoryLength), maxTrajectoryLength(maxTrajectoryLength) {
}

Path::Planner::~Planner() {
}

Path::Trajectory Path::Planner::CalcNext(const VehicleData &vehicleData, const Path::Trajectory &simulatorTrajectory,
                                         double previousEnd_s, double previousEnd_d) {
  /* Adjust internally stored previous trajectory to match non processed parts of the trajectory that was presented to
   * the simulator. */
  AdjustPreviousTrajectory(simulatorTrajectory);

  Path::Trajectory bestTrajectory;
  if (previousTrajectory.size() < minTrajectoryLength) {
    /* Predict trajectories for all other vehicles. */
    const vector<Path::Trajectory> predictions = predict.calc(vehicleData.others, previousTrajectory.size());

    /* Get ego vehicle state at end of previous trajectory. Since there is a more accurate Frenet transformation
     * available from the simulator that one is used instead. */
    VehicleData::EgoVehicleData endState = previousTrajectory.getEndState(vehicleData.ego);
    if (previousTrajectory.size() > 0) {
      endState.s = previousEnd_s;
      endState.d = previousEnd_d;
    }

    /* Get intentions that might be reasonable given current position on the road. */
    vector<Logic::Intention> intentionsToEvaluate = logic.GetIntentionsToEvaluate(endState.d);

    /* Generate a number of trajectories for each intention. */
    vector<Path::Trajectory> trajectories = generateTrajectories(endState, intentionsToEvaluate);

    /* Evaluate generated trajectories using cost functions. */
    double lowestCost = HUGE_VAL;
    for (auto trajectory = trajectories.begin(); trajectory != trajectories.end(); ++trajectory) {
      const double cost = CostCalculator(vehicleData, predictions, *trajectory, false);
      if (cost < lowestCost) {
        lowestCost = cost;
        bestTrajectory = *trajectory;
      }
    }
    cout << "lowest cost = " << CostCalculator(vehicleData, predictions, bestTrajectory, true) << endl;
  }

  Path::Trajectory output = previousTrajectory + bestTrajectory;

  /* To avoid unexpected behavior, it's recommended to not alternate trajectory coordinates that already have been
   * presented to the simulator. To still have the flexibility to react on unexpected changes in the traffic, the
   * number of coordinates presented to the simulator is reduced to maxTrajectoryLength. */
  if (output.size() > maxTrajectoryLength) {
    output.erase(maxTrajectoryLength, output.size() - 1u);
  }
  previousTrajectory = output;
  return output;
}

void Path::Planner::AdjustPreviousTrajectory(const Path::Trajectory &simulatorTrajectory) {
  const int numProcessed = previousTrajectory.size() - simulatorTrajectory.size();
  if (numProcessed > 0) {
    previousTrajectory.erase(0u, numProcessed - 1);
    assert(previousTrajectory.size() == simulatorTrajectory.size());
  }
}

static vector<Path::Trajectory> generateTrajectories(const VehicleData::EgoVehicleData &ego, vector<Path::Logic::Intention> intentionsToEvaluate) {
  const double egoLane = static_cast<int>(Helpers::GetLane(ego.d));
  vector<Path::Trajectory> trajectories;
  for (auto intention = intentionsToEvaluate.begin(); intention != intentionsToEvaluate.end(); ++intention) {
    double delta_d;
    Path::Trajectory trajectory;
    switch (*intention) {
      case Path::Logic::KeepLane:
        trajectory = Path::TrajectoryCalculator::Accelerate(ego, constants.speedLimit - ego.speed);
        trajectory += Path::TrajectoryCalculator::AdjustSpeed(Path::Logic::None, trajectory.getEndState(ego), 120.0 - (trajectory.getEndState(ego).s - ego.s), 0.0, 0.0);
        if (trajectory.size() > 300) {
          trajectory.erase(300, trajectory.size() - 1);
        }
        trajectories.push_back(trajectory);
        delta_d = (egoLane * constants.laneWidth + constants.laneWidth / 2.0) - ego.d;
        break;
      case Path::Logic::LaneChangeLeft:
        delta_d = ((egoLane - 1) * constants.laneWidth + constants.laneWidth / 2.0) - ego.d;
        break;
      case Path::Logic::LaneChangeRight:
        delta_d = ((egoLane + 1) * constants.laneWidth + constants.laneWidth / 2.0) - ego.d;
        break;
      case Path::Logic::PrepareLaneChangeLeft:
        continue; // Not implemented.
      case Path::Logic::PrepareLaneChangeRight:
        continue; // Not implemented.
      case Path::Logic::None:
      case Path::Logic::Unknown:
        assert(false);
    }

    for (double delta_s = 30.0; delta_s < 70.1; delta_s += 10.0) {
      for (double delta_speed = max(-5.0, -ego.speed); delta_speed < min(5.0, constants.speedLimit - ego.speed); delta_speed += 1.0) {
        /* Intended trajectory */
        trajectory = Path::TrajectoryCalculator::AdjustSpeed(*intention, ego, delta_s, delta_d, delta_speed);

        /* Extend trajectory in same lane to make it possible to see how it evolves. */
        trajectory += Path::TrajectoryCalculator::AdjustSpeed(Path::Logic::None, trajectory.getEndState(ego), 120.0 - delta_s, 0.0, 0.0);
        if (trajectory.size() > 300) {
          trajectory.erase(300, trajectory.size() - 1);
        }
        trajectories.push_back(trajectory);
      }
    }
  }
  return trajectories;
}

double Path::Planner::CostCalculator(const VehicleData &vehicleData, const vector<Path::Trajectory> &predictions,
                                     const Path::Trajectory &trajectory, bool verbose) {

  if (trajectory.size() < 10u) {
    return HUGE_VAL;
  }
  double cost = 0.0;

  const VehicleData::EgoVehicleData previousEgoEndState = previousTrajectory.getEndState(vehicleData.ego);
  const VehicleData::EgoVehicleData egoEndState = trajectory.getEndState(previousEgoEndState);

  if (verbose) cout << "ego: " << egoEndState << endl;
  const Path::Trajectory::Kinematics egoKinematics = (previousTrajectory + trajectory).getKinematics();

  vector<VehicleData::EgoVehicleData> othersEndState;
  vector<Path::Trajectory::Kinematics> othersKinematics;
  for (size_t i = 0u; i < vehicleData.others.size(); ++i) {
    assert(predictions[i].size() >= trajectory.size());
    othersEndState.push_back(predictions[i].getState(vehicleData.others[i], trajectory.size() - 1));
    if (verbose) cout << i << ": " << othersEndState.back() << endl;
    othersKinematics.push_back(predictions[i].getKinematics());
  }

  /* Add cost for not keeping to previously not intended decision. */
  const double changedIntentionCost = 1000000.0;
  if ((previousEgoEndState.targetLane != trajectory.targetLane[0]) &&
      (previousEgoEndState.intention != Logic::Intention::None)) {
    if (verbose) cout << "Changing intention from " << previousEgoEndState.intention <<
                         " to " << trajectory.intention[0] << endl;
    cost += changedIntentionCost;
  } else {
    if (verbose) cout << "Sticking to intention " << trajectory.intention[0] <<
                         " with targetLane " << trajectory.targetLane[0] << endl;

  }


  /* Add cost for not keeping speed limit */
  const double slowSpeedCostFactor = 100000.0;
  cost += slowSpeedCostFactor * fabs(constants.speedLimit - egoKinematics.speeds.back());

  /* Add cost when exceeding speed limit */
  const double exceedSpeedLimitCost = 1.0e5;
  cost += (egoKinematics.speeds.back() > constants.speedLimit) ? exceedSpeedLimitCost : 0.0;
  const int startLane = Helpers::GetLane(vehicleData.ego.d);
  const int endLane = trajectory.size() < 2 ? startLane :
                                              Helpers::GetLane(trajectory.x.end()[-1], trajectory.y.end()[-1], Helpers::CalcYaw(trajectory.x.end()[-2], trajectory.y.end()[-2], trajectory.x.end()[-1], trajectory.y.end()[-1]));

  /* Add cost for lane change. */
  const double laneChangeCostFactor = 10000.0;
  cost += laneChangeCostFactor * static_cast<double>(abs(endLane - startLane));

  /* Add cost for driving near other vehicles. */
  double shortestDistance = HUGE_VAL;
  for (auto prediction = predictions.begin(); prediction != predictions.end(); ++prediction) {
    assert(prediction->size() >= trajectory.size());
    for (int i = 0; i < trajectory.size(); ++i) {
      shortestDistance = min(shortestDistance, Helpers::distance(trajectory.x[i], trajectory.y[i], prediction->x[i], prediction->y[i]));
    }
  }
  const double inverseDistanceCostFactor = 10.0;
  cost += inverseDistanceCostFactor / shortestDistance;
  const double collisionCost = 1.0e10;
  const double collisionDistance = 3.0;
  if (verbose) cout << "shortestDistance = " << shortestDistance << endl;
  cost += shortestDistance < collisionDistance ? collisionCost : 0.0;

  /* Add cost for slow vehicles ahead. */
  double slowestSpeedAhead = constants.speedLimit;
  for (int i = 0; i < vehicleData.others.size(); ++i) {
    if (Helpers::GetLane(othersEndState[i].d) == endLane &&
        Helpers::calcLongitudinalDiff(othersEndState[i].s, egoEndState.s) > 0.0) {
      slowestSpeedAhead = min(slowestSpeedAhead, othersEndState[i].speed);
    }
  }
  const double slowSpeedAheadCostFactor = 10000.0;
  cost += (constants.speedLimit - slowestSpeedAhead) * slowSpeedAheadCostFactor;

  /* Add cost for driving close to vehicle ahead. */
  double shortestDistanceAhead = HUGE_VAL;
  for (int i = 0; i < vehicleData.others.size(); ++i) {
    if (Helpers::GetLane(othersEndState[i].d) == endLane) {
      double distanceAhead = Helpers::calcLongitudinalDiff(othersEndState[i].s, egoEndState.s);
      if (distanceAhead >= 0.0) {
        shortestDistanceAhead = min(shortestDistanceAhead, distanceAhead);
      }
    }
  }
  const double longitudinalTimeDiff = shortestDistanceAhead / egoEndState.speed;

  /* In Sweden this is the recommended distance to a vehicle ahead, measured in seconds. */
  const double recommendedLongitudinalTimeDiff = 3.0;
  const double violateRecommendedLongitudinalTimeDiffCost = 1.5e5;
  if (longitudinalTimeDiff < recommendedLongitudinalTimeDiff) {
    if (verbose) cout << "Violated recommended distance" << endl;;
    cost += violateRecommendedLongitudinalTimeDiffCost;
  } else {
    if (verbose) cout << "Not violated recommended distance" << endl;
  }

  const double criticalLongitudinalTimeDiff = 2.0;
  const double violateCriticalLongitudinalTimeDiffCost = 1.5e6;
  if (longitudinalTimeDiff < criticalLongitudinalTimeDiff) {
    if (verbose) cout <<"Violated recommended distance" << endl;
    cost += violateCriticalLongitudinalTimeDiffCost;
  } else {
    if (verbose) cout << "Not violated critical distance" << endl;
  }

  const double accelerationCostFactor = 10000.0;
  auto maxAcceleration = max_element(egoKinematics.accelerations.begin(), egoKinematics.accelerations.end(), fabsCompare);
  cost += fabs(*maxAcceleration) * accelerationCostFactor;

  const double exceedAccelerationLimitCost = 100000.0;
  cost += fabs(*maxAcceleration) > constants.accelerationLimit ? exceedAccelerationLimitCost : 0.0;

  const double jerkCostFactor = 10.0;
  auto maxJerk = max_element(egoKinematics.jerks.begin(), egoKinematics.jerks.end(), fabsCompare);
  cost += fabs(*maxJerk) * jerkCostFactor;

  const double yawRateCostFactor = 100000.0;
  auto maxYawRate = max_element(egoKinematics.yawRates.begin(), egoKinematics.yawRates.end(), fabsCompare);
  cost += fabs(*maxYawRate) * jerkCostFactor;

  if (verbose) cout << "acc=" << fabs(*maxAcceleration) << ", jerk=" << fabs(*maxJerk) << ", yawRate=" << fabs(*maxYawRate) << endl;



  return cost;
}

static bool fabsCompare(double a, double b) {
    return (fabs(a) < fabs(b));
}
