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

TEST_CASE("Path planner should adjust speed to vehicle ahead", "[path_planner]") {
  Path::Planner planner(100);
  const double x = 0.0, y = 0.0, yaw = 0.0, speed = constants.speedLimit;
  double s, d;
  tie(s, d) = helpers.getFrenet(x, y, yaw);
  const vector<vector<double>> otherVehicles = {
    {
      0,
      x + speed * 2.5,
      0.0,
      speed * 0.6,
      0.0,
      s + speed * 2.5,
      d
    }
  };
  const VehicleData vehicleData(x, y, s, d, yaw, speed, otherVehicles);
  const Path::Trajectory previousPath;
  Path::Trajectory nextPath = planner.CalcNext(vehicleData, previousPath);
  Path::Trajectory::Kinematics kinematics = nextPath.getKinematics();
  REQUIRE(kinematics.speeds[0] == Approx(vehicleData.ego.speed).margin(0.02));
  REQUIRE(kinematics.speeds.back() == Approx(vehicleData.others[0].vx).margin(0.15));
}

TEST_CASE("Path planner should adjust speed below vehicle ahead", "[path_planner]") {
  Path::Planner planner(100);
  const double x = 0.0, y = 0.0, yaw = 0.0, speed = constants.speedLimit;
  double s, d;
  tie(s, d) = helpers.getFrenet(x, y, yaw);
  const vector<vector<double>> otherVehicles = {
    {
      0,
      x + speed * 1.9,
      0.0,
      speed * 0.6,
      0.0,
      s + speed * 1.9,
      d
    }
  };
  const VehicleData vehicleData(x, y, s, d, yaw, speed, otherVehicles);
  const Path::Trajectory previousPath;
  Path::Trajectory nextPath = planner.CalcNext(vehicleData, previousPath);
  Path::Trajectory::Kinematics kinematics = nextPath.getKinematics();

  REQUIRE(kinematics.speeds[0] == Approx(vehicleData.ego.speed).margin(0.02));
  REQUIRE(kinematics.speeds.back() < vehicleData.others[0].vx - 0.1);
}
