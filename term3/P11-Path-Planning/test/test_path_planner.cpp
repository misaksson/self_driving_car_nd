#include "catch.hpp"
#include "../src/helpers.h"
#include "../src/path_planner.h"
#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>

using namespace std;

/** Helper function to calculate the speed, acceleration and jerk vectors given a path. */
static tuple<vector<double>, vector<double>, vector<double>> calcSpeedAccJerk(const PathPlanner::Path &path, double deltaTime);

TEST_CASE("Path planner should apply optimal acceleration", "[path_planner]") {
  PathPlanner pathPlanner("../data/test_map.csv", 0, 500);
  double x = 0.0, y = 0.0, yaw = 0.0;
  double s, d;
  tie(s, d) = Helpers::getFrenet(x, y, yaw, pathPlanner.map_waypoints_x, pathPlanner.map_waypoints_y);
  PathPlanner::EgoVehicleData egoVehicle(0.0, 0.0, s, d, 0.0, 0.0);
  vector<PathPlanner::OtherVehicleData> otherVehicles;
  PathPlanner::Path previousPath;
  PathPlanner::Path nextPath = pathPlanner.CalcNext(egoVehicle, otherVehicles, previousPath, s, d);

  vector<double> speeds, accelerations, jerks;
  tie(speeds, accelerations, jerks) = calcSpeedAccJerk(nextPath, pathPlanner.deltaTime);

  REQUIRE(*min_element(speeds.begin(), speeds.end()) == Approx(0.01).margin(0.01));
  REQUIRE(*max_element(speeds.begin(), speeds.end()) == Approx(pathPlanner.speedLimit).margin(0.001));
  REQUIRE(*min_element(accelerations.begin(), accelerations.end()) == Approx(0.0).margin(0.01));
  REQUIRE(*max_element(accelerations.begin(), accelerations.end()) == Approx(pathPlanner.accelerationLimit).margin(0.001));
  REQUIRE(*min_element(jerks.begin(), jerks.end()) == Approx(-pathPlanner.jerkLimit).margin(0.001));
  REQUIRE(*max_element(jerks.begin(), jerks.end()) == Approx(pathPlanner.jerkLimit).margin(0.001));

  enum AccelerationPhase {
    IncreasingAcceleration = 0,
    ConstantMaxAcceleration,
    DecreasingAcceleration,
    ConstantZeroAcceleration,
  };
  AccelerationPhase accelerationPhase = IncreasingAcceleration;
  for (int i = 0; i < jerks.size(); ++i) {
    switch (accelerationPhase) {
      case IncreasingAcceleration: // From stand still.
        if (accelerations[i + 1] < pathPlanner.accelerationLimit - 0.01) {
          REQUIRE(jerks[i] == Approx(pathPlanner.jerkLimit).margin(0.001));
        } else {
          double expectedSpeed = pow(pathPlanner.accelerationLimit, 2.0) / (2.0 * pathPlanner.jerkLimit);
          REQUIRE(speeds[i + 2] == Approx(expectedSpeed).margin(0.1));
          accelerationPhase = ConstantMaxAcceleration;
        }
        break;
      case ConstantMaxAcceleration:
        if (accelerations[i + 1] > pathPlanner.accelerationLimit - 0.01) {
          REQUIRE(jerks[i] == Approx(0.0).margin(0.001));
        } else {
          double expectedSpeed = pathPlanner.speedLimit - pow(pathPlanner.accelerationLimit, 2.0) / (2.0 * pathPlanner.jerkLimit);
          REQUIRE(speeds[i + 2] == Approx(expectedSpeed).margin(0.3));
          accelerationPhase = DecreasingAcceleration;
        }
        break;
      case DecreasingAcceleration:
        if (accelerations[i + 1] > 0.01) {
          REQUIRE(jerks[i] == Approx(-pathPlanner.jerkLimit).margin(0.001));
        } else {
          accelerationPhase = ConstantZeroAcceleration;
        }
        break;
      case ConstantZeroAcceleration:
        REQUIRE(jerks[i] == Approx(0.0).margin(0.4));
        REQUIRE(accelerations[i + 1] == Approx(0.0).margin(0.004));
        REQUIRE(speeds[i + 2] == Approx(pathPlanner.speedLimit).margin(0.001));
        break;
    }
  }
}

TEST_CASE("Path planner should adjust speed to vehicle ahead", "[path_planner]") {
  PathPlanner pathPlanner("../data/test_map.csv", 0, 100);
  double x = 0.0, y = 0.0, yaw = 0.0;
  double s, d;
  tie(s, d) = Helpers::getFrenet(x, y, yaw, pathPlanner.map_waypoints_x, pathPlanner.map_waypoints_y);
  PathPlanner::EgoVehicleData egoVehicle(0.0, 0.0, s, d, 0.0, pathPlanner.speedLimit);
  vector<PathPlanner::OtherVehicleData> otherVehicles = {
    PathPlanner::OtherVehicleData({0,
                                   egoVehicle.x + egoVehicle.speed * 2.5,
                                   0.0,
                                   pathPlanner.speedLimit * 0.6,
                                   0.0,
                                   egoVehicle.s + egoVehicle.speed * 2.5,
                                   egoVehicle.d})
  };
  PathPlanner::Path previousPath;
  PathPlanner::Path nextPath = pathPlanner.CalcNext(egoVehicle, otherVehicles, previousPath, s, d);

  vector<double> speeds, accelerations, jerks;
  tie(speeds, accelerations, jerks) = calcSpeedAccJerk(nextPath, pathPlanner.deltaTime);
  REQUIRE(speeds[0] == Approx(egoVehicle.speed).margin(0.02));
  REQUIRE(speeds.back() == Approx(otherVehicles[0].vx).margin(0.001));
}

TEST_CASE("Path planner should adjust speed below vehicle ahead", "[path_planner]") {
  PathPlanner pathPlanner("../data/test_map.csv", 0, 100);
  double x = 0.0, y = 0.0, yaw = 0.0;
  double s, d;
  tie(s, d) = Helpers::getFrenet(x, y, yaw, pathPlanner.map_waypoints_x, pathPlanner.map_waypoints_y);
  PathPlanner::EgoVehicleData egoVehicle(0.0, 0.0, s, d, 0.0, pathPlanner.speedLimit);
  vector<PathPlanner::OtherVehicleData> otherVehicles = {
    PathPlanner::OtherVehicleData({0,
                                   egoVehicle.x + egoVehicle.speed * 1.9,
                                   0.0,
                                   pathPlanner.speedLimit * 0.6,
                                   0.0,
                                   egoVehicle.s + egoVehicle.speed * 1.9,
                                   egoVehicle.d})
  };
  PathPlanner::Path previousPath;
  PathPlanner::Path nextPath = pathPlanner.CalcNext(egoVehicle, otherVehicles, previousPath, s, d);

  vector<double> speeds, accelerations, jerks;
  tie(speeds, accelerations, jerks) = calcSpeedAccJerk(nextPath, pathPlanner.deltaTime);
  REQUIRE(speeds[0] == Approx(egoVehicle.speed).margin(0.02));
  REQUIRE(speeds.back() < otherVehicles[0].vx - 0.1);
}

static tuple<vector<double>, vector<double>, vector<double>> calcSpeedAccJerk(const PathPlanner::Path &path, double deltaTime) {
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
