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
#include <math.h>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace std;

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

  Path::Trajectory output;
  if (previousTrajectory.size() < minTrajectoryLength) {
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
    vector<Path::Trajectory> trajectories = GenerateTrajectories(endState, intentionsToEvaluate);
    Path::Trajectory bestTrajectory = EvaluateTrajectories(vehicleData, trajectories);
    output = previousTrajectory + bestTrajectory;

    /* To avoid unexpected behavior, it's recommended to not alternate trajectory coordinates that already have been
     * presented to the simulator. To still have the flexibility to react on unexpected changes in the traffic, the
     * number of coordinates presented to the simulator is reduced to maxTrajectoryLength. */
    if (output.size() > maxTrajectoryLength) {
      output.erase(maxTrajectoryLength, output.size() - 1u);
    }
  } else {
    output = previousTrajectory;
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

vector<Path::Trajectory> Path::Planner::GenerateTrajectories(const VehicleData::EgoVehicleData &ego, vector<Path::Logic::Intention> intentionsToEvaluate) {
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

Path::Trajectory Path::Planner::EvaluateTrajectories(const VehicleData &vehicleData, const vector<Path::Trajectory> &trajectories) {
  /* Predict trajectories for all other vehicles. */
  const vector<Path::Trajectory> predictions = predict.calc(vehicleData.others, previousTrajectory.size());

  /* Evaluate generated trajectories using cost functions. */
  const double slowSpeedCostFactor = 1.0e5;
  const double exceedSpeedLimitCost = 1.0e5;
  const double changeIntentionCost = 1.0e6;
  const double laneChangeCostFactor = 1.0e4;
  const double inverseDistanceCostFactor = 1.0e1;
  const double collisionCost = 1.0e20;
  const double slowLaneCostFactor = 1.0e4;
  const double violateRecommendedLongitudinalTimeDiffCost = 1.0e6;
  const double violateCriticalLongitudinalTimeDiffCost = 1.0e7;
  const double accelerationCostFactor = 1.0e4;
  const double jerkCostFactor = 1.0e1;
  const double yawRateCostFactor = 1.0e5;
  const double exceedAccelerationLimitCost = 1.0e5;

  vector<unique_ptr<Cost>> costFunctions;
  costFunctions.emplace_back(unique_ptr<Cost>(new SlowSpeed(slowSpeedCostFactor)));
  costFunctions.emplace_back(unique_ptr<Cost>(new ExceedSpeedLimit(exceedSpeedLimitCost)));
  costFunctions.emplace_back(unique_ptr<Cost>(new ChangeIntention(changeIntentionCost)));
  costFunctions.emplace_back(unique_ptr<Cost>(new LaneChange(laneChangeCostFactor)));
  costFunctions.emplace_back(unique_ptr<Cost>(new NearOtherVehicles(inverseDistanceCostFactor)));
  costFunctions.emplace_back(unique_ptr<Cost>(new Collision(collisionCost)));
  costFunctions.emplace_back(unique_ptr<Cost>(new SlowLane(slowLaneCostFactor)));
  costFunctions.emplace_back(unique_ptr<Cost>(new ViolateRecommendedDistanceAhead(violateRecommendedLongitudinalTimeDiffCost)));
  costFunctions.emplace_back(unique_ptr<Cost>(new ViolateCriticalDistanceAhead(violateCriticalLongitudinalTimeDiffCost)));
  costFunctions.emplace_back(unique_ptr<Cost>(new Acceleration(accelerationCostFactor)));
  costFunctions.emplace_back(unique_ptr<Cost>(new Jerk(jerkCostFactor)));
  costFunctions.emplace_back(unique_ptr<Cost>(new YawRate(yawRateCostFactor)));
  costFunctions.emplace_back(unique_ptr<Cost>(new ExceedAccelerationLimit(exceedAccelerationLimitCost)));

  Path::Cost::verbose = false;
  Path::Cost::preprocessCommonData(previousTrajectory, vehicleData, predictions);

  Path::Trajectory bestTrajectory;
  double lowestCost = HUGE_VAL;
  for (auto trajectory = trajectories.begin(); trajectory != trajectories.end(); ++trajectory) {
    if (trajectory->size() > 10) {
      Path::Cost::preprocessCurrentTrajectory(*trajectory);

      double cost = 0.0;
      for (auto const& costFunction: costFunctions) {
        cost += costFunction->calc(*trajectory);
      }
      if (cost < lowestCost) {
        lowestCost = cost;
        bestTrajectory = *trajectory;
      }
    }
  }
  Path::Cost::verbose = true;
  Path::Cost::preprocessCurrentTrajectory(bestTrajectory);
  double cost = 0.0;
  for (auto const& costFunction: costFunctions) {
    cost += costFunction->calc(bestTrajectory);
  }
  cout << "lowest cost = " << cost << endl;
  return bestTrajectory;
}

