#include "../constants.h"
#include "../helpers.h"
#include "../spline.h"
#include "cost.h"
#include "logic.h"
#include "planner.h"
#include "trajectory.h"
#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <math.h>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace std;

Path::Planner::Planner(int minTrajectoryLength, int minUpdatePeriod) :
    logic(Path::Logic()), predict(Path::Predict()), numProcessedSinceLastUpdate(INT_MAX), simulatorTime(0.0),
    minTrajectoryLength(minTrajectoryLength), minUpdatePeriod(minUpdatePeriod) {
}

Path::Planner::~Planner() {
}

Path::Trajectory Path::Planner::CalcNext(const VehicleData &vehicleData, const Path::Trajectory &simulatorTrajectory) {
  /* Adjust internally stored previous trajectory to match non processed parts of the trajectory that was presented to
   * the simulator. */
  AdjustPreviousTrajectory(simulatorTrajectory);

  Path::Trajectory output;
  if (numProcessedSinceLastUpdate > minUpdatePeriod) {
    /* Update trajectory. */
    numProcessedSinceLastUpdate = 0;

    /* Reduced previous trajectory to minimum needed by simulator. */
    if (previousTrajectory.size() > minTrajectoryLength) {
      previousTrajectory.erase(minTrajectoryLength, previousTrajectory.size() - 1);
    }
    /* Get ego vehicle state at end of previous trajectory. Since there is a more accurate Frenet transformation
     * available from the simulator that one is used instead. */
    VehicleData::EgoVehicleData endState = previousTrajectory.getEndState(vehicleData.ego);

    /* Get intentions that might be reasonable given current position on the road. */
    vector<Logic::Intention> intentionsToEvaluate = logic.GetIntentionsToEvaluate(endState.d);

    /* Generate a number of trajectories for each intention. */
    vector<Path::Trajectory> trajectories = GenerateTrajectories(endState, intentionsToEvaluate);
    previousTrajectory += EvaluateTrajectories(vehicleData, trajectories);
  }
  cout << "Simulator time = " << simulatorTime << endl;
  return previousTrajectory;
}

void Path::Planner::AdjustPreviousTrajectory(const Path::Trajectory &simulatorTrajectory) {
  const int numProcessed = previousTrajectory.size() - simulatorTrajectory.size();
  simulatorTime += constants.deltaTime * numProcessed;
  numProcessedSinceLastUpdate += numProcessed;
  if (numProcessed > 0) {
    previousTrajectory.erase(0u, numProcessed - 1);
  }
}

vector<Path::Trajectory> Path::Planner::GenerateTrajectories(const VehicleData::EgoVehicleData &ego, vector<Path::Logic::Intention> intentionsToEvaluate) {
  const double egoLane = static_cast<int>(Helpers::GetLane(ego.d));
  vector<Path::Trajectory> trajectories;
  for (auto intention = intentionsToEvaluate.begin(); intention != intentionsToEvaluate.end(); ++intention) {
    // Each intention specifies a target delta d value, e.g. the change needed to go to target lane.
    double delta_d;

    // Each intention specifies a range of longitudinal distances that is reasonable to adjust for delta_d.
    double minAdjustmentDistance;
    double maxAdjustmentDistance;
    Path::Trajectory trajectory;
    switch (*intention) {
      case Logic::Intention::KeepLane:
        trajectory = Path::TrajectoryCalculator::Accelerate(ego, constants.speedLimit - ego.speed);
        trajectory += Path::TrajectoryCalculator::AdjustSpeed(Path::Logic::KeepLane, trajectory.getEndState(ego), 120.0 - (trajectory.getEndState(ego).s - ego.s), 0.0, 0.0);
        if (trajectory.size() > 300) {
          trajectory.erase(300, trajectory.size() - 1);
        }
        trajectories.push_back(trajectory);
        delta_d = (egoLane * constants.laneWidth + constants.laneWidth / 2.0) - ego.d;
        minAdjustmentDistance = 30.0;
        maxAdjustmentDistance = 40.1;
        break;
      case Logic::Intention::LaneChangeLeft:
        delta_d = ((egoLane - 1) * constants.laneWidth + constants.laneWidth / 2.0) - ego.d;
        minAdjustmentDistance = 30.0;
        maxAdjustmentDistance = 60.1;
        break;
      case Logic::Intention::LaneChangeRight:
        delta_d = ((egoLane + 1) * constants.laneWidth + constants.laneWidth / 2.0) - ego.d;
        minAdjustmentDistance = 30.0;
        maxAdjustmentDistance = 60.1;
        break;
      case Logic::Intention::TwoLaneChangesLeft:
        delta_d = ((egoLane - 2) * constants.laneWidth + constants.laneWidth / 2.0) - ego.d;
        minAdjustmentDistance = 50.0;
        maxAdjustmentDistance = 100.1;
        break;
      case Logic::Intention::TwoLaneChangesRight:
        delta_d = ((egoLane + 2) * constants.laneWidth + constants.laneWidth / 2.0) - ego.d;
        minAdjustmentDistance = 50.0;
        maxAdjustmentDistance = 100.1;
        break;
      case Logic::Intention::PrepareLaneChangeLeft:
        continue; // Not implemented.
      case Logic::Intention::PrepareLaneChangeRight:
        continue; // Not implemented.
      case Logic::Intention::None:
      case Logic::Intention::Unknown:
        assert(false);
    }

    for (double delta_s = minAdjustmentDistance; delta_s < maxAdjustmentDistance; delta_s += 10.0) {
      for (double delta_speed = max(-20.0, -ego.speed); delta_speed < min(20.0, constants.speedLimit - ego.speed); delta_speed += 1.0) {
        /* Intended trajectory */
        trajectory = Path::TrajectoryCalculator::AdjustSpeed(*intention, ego, delta_s, delta_d, delta_speed);

        /* Extend trajectory in same lane to make it possible to see how it evolves. */
        trajectory += Path::TrajectoryCalculator::AdjustSpeed(Path::Logic::KeepLane, trajectory.getEndState(ego), 120.0 - delta_s, 0.0, 0.0);
        if (trajectory.size() > 300) {
          trajectory.erase(300, trajectory.size() - 1);
        }
        trajectories.push_back(trajectory);
      }
    }
  }
  return trajectories;
}

Path::Trajectory Path::Planner::EvaluateTrajectories(const VehicleData &vehicleData, const vector<Path::Trajectory> &trajectories) {
  cout << "\nnum trajectories = " << trajectories.size() << endl;

  /* Predict trajectories for all other vehicles. */
  const vector<Path::Trajectory> predictions = predict.calc(vehicleData.others, previousTrajectory.size());

  /* Evaluate generated trajectories using cost functions. */
  const double slowSpeedCostFactor = 2.0e7;
  const double exceedSpeedLimitCost = 1.0e5;
  const double changeIntentionCost = 1.0e8;
  const double laneChangeCostFactor = 1.0e4;
  const double laneChangeInFrontOfOtherCost = 1.0e9;
  const double laneChangeInFrontOfOtherFasterCost = 1.0e9;
  const double inverseDistanceCostFactor = 1.0e5;
  const double collisionCost = 1.0e10;
  const double slowLaneCostFactor = 1.5e7;
  const double violateCriticalLongitudinalTimeDiffCost = 1.0e9;
  const double accelerationCostFactor = 1.0e4;
  const double jerkCostFactor = 1.0e1;
  const double yawRateCostFactor = 1.0e5;
  const double exceedAccelerationLimitCost = 1.0e5;

  vector<unique_ptr<Cost>> costFunctions;
  costFunctions.emplace_back(unique_ptr<Cost>(new SlowSpeed(slowSpeedCostFactor)));
  costFunctions.emplace_back(unique_ptr<Cost>(new ExceedSpeedLimit(exceedSpeedLimitCost)));
  costFunctions.emplace_back(unique_ptr<Cost>(new ChangeIntention(changeIntentionCost)));
  costFunctions.emplace_back(unique_ptr<Cost>(new LaneChange(laneChangeCostFactor)));
  costFunctions.emplace_back(unique_ptr<Cost>(new LaneChangeInFrontOfOther(laneChangeInFrontOfOtherCost)));
  costFunctions.emplace_back(unique_ptr<Cost>(new LaneChangeInFrontOfOtherFaster(laneChangeInFrontOfOtherFasterCost)));
  costFunctions.emplace_back(unique_ptr<Cost>(new NearOtherVehicles(inverseDistanceCostFactor)));
  costFunctions.emplace_back(unique_ptr<Cost>(new Collision(collisionCost)));
  costFunctions.emplace_back(unique_ptr<Cost>(new SlowLane(slowLaneCostFactor)));
  costFunctions.emplace_back(unique_ptr<Cost>(new ViolateCriticalDistanceAhead(violateCriticalLongitudinalTimeDiffCost)));
  costFunctions.emplace_back(unique_ptr<Cost>(new Acceleration(accelerationCostFactor)));
  costFunctions.emplace_back(unique_ptr<Cost>(new Jerk(jerkCostFactor)));
  costFunctions.emplace_back(unique_ptr<Cost>(new YawRate(yawRateCostFactor)));
  costFunctions.emplace_back(unique_ptr<Cost>(new ExceedAccelerationLimit(exceedAccelerationLimitCost)));

  Path::Cost::verbose = false;
  Path::Cost::preprocessCommonData(previousTrajectory, vehicleData, predictions);

  double lowestCost = HUGE_VAL;
  int bestIdx = 0;
  for (int i = 0; i < trajectories.size(); ++i) {
    if (Path::Cost::verbose) cout << "IDX " << i << endl;
    if (trajectories[i].size() > 10) {
      Path::Cost::preprocessCurrentTrajectory(trajectories[i]);

      double cost = 0.0;
      for (auto const& costFunction: costFunctions) {
        cost += costFunction->getCost(trajectories[i]);
      }
      if (Path::Cost::verbose) cout << "Total cost of trajectory = " << cost << endl;
      if (cost < lowestCost) {
        lowestCost = cost;
        bestIdx = i;
      }
    }
  }
  if (Path::Cost::verbose) cout << "BEST TRAJECTORY" << endl;
  Path::Cost::verbose = true;
  Path::Cost::preprocessCurrentTrajectory(trajectories[bestIdx]);
  double cost = 0.0;
  for (auto const& costFunction: costFunctions) {
    cost += costFunction->getCost(trajectories[bestIdx]);
  }
  cout << "Lowest cost " << cost << " was calculated for trajectory " << bestIdx << endl;
  return trajectories[bestIdx];
}

