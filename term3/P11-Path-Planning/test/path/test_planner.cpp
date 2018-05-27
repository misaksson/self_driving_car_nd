#include "../../src/constants.h"
#include "../../src/helpers.h"
#include "../../src/path/planner.h"
#include "../../src/vehicle_data.h"
#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>
#include "../catch.hpp"

using namespace std;

TEST_CASE("Path planner should adjust speed to vehicle ahead", "[path]") {
  Path::Planner planner(100, 400);
  const double x = 0.0, y = -6.0, yaw = 0.0, speed = constants.speedLimit;
  double s, d;
  tie(s, d) = helpers.getFrenet(x, y, yaw);
  const vector<vector<double>> otherVehicles = {
    // Vehicle in same lane keeping highest speed.
    {0, x + speed * 2.5, y, speed * 0.6, 0.0, s + speed * 2.5, d},

    /* Block the other lanes with even slower vehicles. */
    {1, x + speed * 2.5, y - constants.laneWidth, speed * 0.59, 0.0, s + speed * 2.5, d + constants.laneWidth},
    {2, x + speed * 2.5, y + constants.laneWidth, speed * 0.59, 0.0, s + speed * 2.5, d - constants.laneWidth},
  };
  const VehicleData vehicleData(x, y, s, d, yaw, speed, otherVehicles);
  const Path::Trajectory previousPath;
  Path::Trajectory trajectory = planner.CalcNext(vehicleData, previousPath);
  Path::Trajectory::Kinematics kinematics = trajectory.getKinematics();
  VehicleData::EgoVehicleData endState = trajectory.getEndState(vehicleData.ego);
  REQUIRE(kinematics.speeds[0] == Approx(vehicleData.ego.speed).margin(0.5));
  REQUIRE(kinematics.speeds.back() == Approx(vehicleData.others[0].speed).margin(1.0));
  REQUIRE(endState.d == Approx(d).margin(0.1));
}

TEST_CASE("Path planner should adjust speed below vehicle ahead", "[path]") {
  Path::Planner planner(100, 400);
  const double x = 0.0, y = -6.0, yaw = 0.0, speed = constants.speedLimit;
  double s, d;
  tie(s, d) = helpers.getFrenet(x, y, yaw);
  const vector<vector<double>> otherVehicles = {
    // Vehicle in same lane keeping highest speed.
    {0, x + speed * 1.2, y, speed * 0.6, 0.0, s + speed * 2.5, d},

    /* Block the other lanes with even slower vehicles. */
    {1, x + speed * 1.2, y - constants.laneWidth, speed * 0.59, 0.0, s + speed * 1.2, d + constants.laneWidth},
    {2, x + speed * 1.2, y + constants.laneWidth, speed * 0.59, 0.0, s + speed * 1.2, d - constants.laneWidth},
  };
  const VehicleData vehicleData(x, y, s, d, yaw, speed, otherVehicles);
  const Path::Trajectory previousPath;
  Path::Trajectory trajectory = planner.CalcNext(vehicleData, previousPath);
  Path::Trajectory::Kinematics kinematics = trajectory.getKinematics();
  VehicleData::EgoVehicleData endState = trajectory.getEndState(vehicleData.ego);
  REQUIRE(kinematics.speeds[0] == Approx(vehicleData.ego.speed).margin(0.5));
  REQUIRE(kinematics.speeds.back() < vehicleData.others[0].speed - 0.1);
  REQUIRE(endState.d == Approx(d).margin(0.1));
}

TEST_CASE("Path planner should switch lane left", "[path]") {
  Path::Planner planner(100, 400);
  const double x = 0.0, y = -6.0, yaw = 0.0, speed = constants.speedLimit;
  double s, d;
  tie(s, d) = helpers.getFrenet(x, y, yaw);
  const vector<vector<double>> otherVehicles = {
    // Vehicle ahead in same lane going slow.
    {0, x + speed * 1.5, y, speed * 0.6, 0.0, s + speed * 1.5, d},
    // Vehicle ahead in right lane going slow.
    {1, x + speed * 1.5, y - constants.laneWidth, speed * 0.6, 0.0, s + speed * 1.5, d + constants.laneWidth},
  };
  const VehicleData vehicleData(x, y, s, d, yaw, speed, otherVehicles);
  const Path::Trajectory previousPath;
  Path::Trajectory trajectory = planner.CalcNext(vehicleData, previousPath);
  Path::Trajectory::Kinematics kinematics = trajectory.getKinematics();
  VehicleData::EgoVehicleData endState = trajectory.getEndState(vehicleData.ego);
  REQUIRE(kinematics.speeds[0] == Approx(vehicleData.ego.speed).margin(0.5));
  REQUIRE(kinematics.speeds.back() == Approx(vehicleData.ego.speed).margin(0.5));
  REQUIRE(endState.d == Approx(2).margin(0.1));
}
