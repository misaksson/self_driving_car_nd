#include "../src/constants.h"
#include "../src/helpers.h"
#include "../src/path_planner.h"
#include "../src/vehicle_data.h"
#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>
#include "catch.hpp"

using namespace std;

/** Helper function to calculate the speed, acceleration and jerk vectors given a path. */
static tuple<vector<double>, vector<double>, vector<double>> calcSpeedAccJerk(const Path::Trajectory &trajectory, double deltaTime);

TEST_CASE("Path planner should adjust speed to vehicle ahead", "[path_planner]") {
  Helpers helpers("../data/test_map.csv");
  Path::TrajectoryCalculator trajectoryCalculator(helpers);
  PathPlanner pathPlanner(helpers, trajectoryCalculator, 100);
  const double x = 0.0, y = 0.0, yaw = 0.0;
  double s, d;
  tie(s, d) = helpers.getFrenet(x, y, yaw);
  const VehicleData::EgoVehicleData egoVehicle(0.0, 0.0, s, d, 0.0, constants.speedLimit);
  const vector<VehicleData::OtherVehicleData> otherVehicles = {
    VehicleData::OtherVehicleData({0,
                                   egoVehicle.x + egoVehicle.speed * 2.5,
                                   0.0,
                                   constants.speedLimit * 0.6,
                                   0.0,
                                   egoVehicle.s + egoVehicle.speed * 2.5,
                                   egoVehicle.d})
  };
  const VehicleData vehicleData(egoVehicle, otherVehicles);
  const Path::Trajectory previousPath;
  Path::Trajectory nextPath = pathPlanner.CalcNext(vehicleData, previousPath, s, d);

  vector<double> speeds, accelerations, jerks;
  tie(speeds, accelerations, jerks) = calcSpeedAccJerk(nextPath, constants.deltaTime);
  REQUIRE(speeds[0] == Approx(egoVehicle.speed).margin(0.02));
  REQUIRE(speeds.back() == Approx(otherVehicles[0].vx).margin(0.15));
}

TEST_CASE("Path planner should adjust speed below vehicle ahead", "[path_planner]") {
  Helpers helpers("../data/test_map.csv");
  Path::TrajectoryCalculator trajectoryCalculator(helpers);
  PathPlanner pathPlanner(helpers, trajectoryCalculator, 100);
  const double x = 0.0, y = 0.0, yaw = 0.0;
  double s, d;
  tie(s, d) = helpers.getFrenet(x, y, yaw);
  const VehicleData::EgoVehicleData egoVehicle(0.0, 0.0, s, d, 0.0, constants.speedLimit);
  const vector<VehicleData::OtherVehicleData> otherVehicles = {
    VehicleData::OtherVehicleData({0,
                                   egoVehicle.x + egoVehicle.speed * 1.9,
                                   0.0,
                                   constants.speedLimit * 0.6,
                                   0.0,
                                   egoVehicle.s + egoVehicle.speed * 1.9,
                                   egoVehicle.d})
  };
  const VehicleData vehicleData(egoVehicle, otherVehicles);
  const Path::Trajectory previousPath;
  Path::Trajectory nextPath = pathPlanner.CalcNext(vehicleData, previousPath, s, d);

  vector<double> speeds, accelerations, jerks;
  tie(speeds, accelerations, jerks) = calcSpeedAccJerk(nextPath, constants.deltaTime);
  REQUIRE(speeds[0] == Approx(egoVehicle.speed).margin(0.02));
  REQUIRE(speeds.back() < otherVehicles[0].vx - 0.1);
}

static tuple<vector<double>, vector<double>, vector<double>> calcSpeedAccJerk(const Path::Trajectory &path, double deltaTime) {
  vector<double> speeds, accelerations, jerks;
  for (int i = 1; i < path.x.size(); ++i) {
    speeds.push_back(Helpers::distance(path.x[i - 1], path.y[i - 1], path.x[i], path.y[i]) / deltaTime);
  }
  for (int i = 1; i < speeds.size(); ++i) {
    accelerations.push_back((speeds[i] - speeds[i - 1]) / deltaTime);
  }
  for (int i = 1; i < accelerations.size(); ++i) {
    jerks.push_back((accelerations[i] - accelerations[i - 1]) / deltaTime);
  }
  return make_tuple(speeds, accelerations, jerks);
}
