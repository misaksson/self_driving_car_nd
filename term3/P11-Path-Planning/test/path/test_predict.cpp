#include "../../src/constants.h"
#include "../../src/helpers.h"
#include "../../src/path/predict.h"
#include "../../src/vehicle_data.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include "../catch.hpp"

using namespace std;

TEST_CASE("Should predict keep lane", "[path]") {
  Path::Predict predict;
  const double x = 100.0, y = -6.0, vx = 20.0, vy = 0.0, yaw = 0.0;
  double s, d;
  tie(s, d) = helpers.getFrenet(x, y, yaw);
  const vector<vector<double>> otherVehicles = {
    {0, x, y, vx, vy, s, d}
  };
  const VehicleData vehicleData(0, 0, 0, 0, 0, 0, otherVehicles);
  vector<Path::Trajectory> trajectories = predict.calc(vehicleData.others, 0);

  Path::Trajectory::Kinematics kinematics = trajectories[0].getKinematics();
  VehicleData::EgoVehicleData endState = trajectories[0].getEndState(vehicleData.ego);

  REQUIRE(endState.d == Approx(d).margin(0.5));
  REQUIRE(kinematics.speeds[0] == Approx(vx).margin(0.5));
  REQUIRE(kinematics.speeds.back() == Approx(vx).margin(0.5));
}

TEST_CASE("Should predict lane change left", "[path]") {
  Path::Predict predict;
  const double x = 100.0, y = -5.5, vx = 20.0, vy = 3, yaw = 0.0;
  double s, d;
  tie(s, d) = helpers.getFrenet(x, y, yaw);
  const vector<vector<double>> otherVehicles = {
    {0, x, y, vx, vy, s, d}
  };
  const VehicleData vehicleData(0, 0, 0, 0, 0, 0, otherVehicles);
  vector<Path::Trajectory> trajectories = predict.calc(vehicleData.others, 0);

  Path::Trajectory::Kinematics kinematics = trajectories[0].getKinematics();
  VehicleData::EgoVehicleData endState = trajectories[0].getEndState(vehicleData.ego);

  REQUIRE(endState.d == Approx(2.0).margin(0.5));
  REQUIRE(kinematics.speeds[0] == Approx(vx).margin(0.5));
  REQUIRE(kinematics.speeds.back() == Approx(vx).margin(0.5));
}

TEST_CASE("Should predict lane change right", "[path]") {
  Path::Predict predict;
  const double x = 100.0, y = -6.5, vx = 20.0, vy = -3, yaw = 0.0;
  double s, d;
  tie(s, d) = helpers.getFrenet(x, y, yaw);
  const vector<vector<double>> otherVehicles = {
    {0, x, y, vx, vy, s, d}
  };
  const VehicleData vehicleData(0, 0, 0, 0, 0, 0, otherVehicles);
  vector<Path::Trajectory> trajectories = predict.calc(vehicleData.others, 0);

  Path::Trajectory::Kinematics kinematics = trajectories[0].getKinematics();
  VehicleData::EgoVehicleData endState = trajectories[0].getEndState(vehicleData.ego);

  REQUIRE(endState.d == Approx(10.0).margin(0.5));
  REQUIRE(kinematics.speeds[0] == Approx(vx).margin(0.5));
  REQUIRE(kinematics.speeds.back() == Approx(vx).margin(0.5));
}

