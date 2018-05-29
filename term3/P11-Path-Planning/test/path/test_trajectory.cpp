#include "../../src/constants.h"
#include "../../src/helpers.h"
#include "../../src/vehicle_data.h"
#include "../../src/path/trajectory.h"
#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>
#include "../catch.hpp"

using namespace std;

TEST_CASE("Trajectory should find optimal acceleration", "[path]") {
  double x = 100.0, y = 0.0, yaw = 0.0;
  double s, d;
  tie(s, d) = helpers.getFrenet(x, y, yaw);

  vector<vector<double>> sensorFusion;
  const VehicleData vehicleData(x, y, s, d, yaw, 0.0, sensorFusion);
  Path::Trajectory trajectory = Path::TrajectoryCalculator::Accelerate(vehicleData.ego, constants.speedLimit);

  Path::Trajectory::Kinematics kinematics = trajectory.getKinematics();
  REQUIRE(*min_element(kinematics.speeds.begin(), kinematics.speeds.end()) == Approx(0.01).margin(0.01));
  REQUIRE(*max_element(kinematics.speeds.begin(), kinematics.speeds.end()) == Approx(constants.speedLimit).margin(0.1));
  REQUIRE(*min_element(kinematics.accelerations.begin(), kinematics.accelerations.end()) >= 0.0);
  REQUIRE(*max_element(kinematics.accelerations.begin(), kinematics.accelerations.end()) == Approx(constants.accelerationLimit).margin(0.001));
  REQUIRE(*min_element(kinematics.jerks.begin(), kinematics.jerks.end()) == Approx(-constants.jerkLimit).margin(0.001));
  REQUIRE(*max_element(kinematics.jerks.begin(), kinematics.jerks.end()) == Approx(constants.jerkLimit).margin(0.001));

  enum AccelerationPhase {
    IncreasingAcceleration = 0,
    ConstantMaxAcceleration,
    DecreasingAcceleration,
    ConstantZeroAcceleration,
  };
  AccelerationPhase accelerationPhase = IncreasingAcceleration;
  for (int i = 0; i < kinematics.jerks.size(); ++i) {
    switch (accelerationPhase) {
      case IncreasingAcceleration: // From stand still.
        if (kinematics.accelerations[i + 1] < constants.accelerationLimit - 0.01) {
          REQUIRE(kinematics.jerks[i] == Approx(constants.jerkLimit).margin(0.001));
        } else {
          double expectedSpeed = pow(constants.accelerationLimit, 2.0) / (2.0 * constants.jerkLimit);
          REQUIRE(kinematics.speeds[i + 2] == Approx(expectedSpeed).margin(0.1));
          accelerationPhase = ConstantMaxAcceleration;
        }
        break;
      case ConstantMaxAcceleration:
        if (kinematics.accelerations[i + 1] > constants.accelerationLimit - 0.01) {
          REQUIRE(kinematics.jerks[i] == Approx(0.0).margin(0.001));
        } else {
          double expectedSpeed = constants.speedLimit - pow(constants.accelerationLimit, 2.0) / (2.0 * constants.jerkLimit);
          REQUIRE(kinematics.speeds[i + 2] == Approx(expectedSpeed).margin(0.3));
          accelerationPhase = DecreasingAcceleration;
        }
        break;
      case DecreasingAcceleration:
        if (kinematics.accelerations[i + 1] > 0.01) {
          REQUIRE(kinematics.jerks[i] == Approx(-constants.jerkLimit).margin(0.001));
        } else {
          accelerationPhase = ConstantZeroAcceleration;
        }
        break;
      case ConstantZeroAcceleration:
        REQUIRE(kinematics.jerks[i] == Approx(0.0).margin(0.4));
        REQUIRE(kinematics.accelerations[i + 1] == Approx(0.0).margin(0.004));
        REQUIRE(kinematics.speeds[i + 2] == Approx(constants.speedLimit).margin(0.001));
        break;
    }
  }
}

TEST_CASE("Trajectory should smoothly get from A to B", "[path]") {
  const double x = 100.0, y = 0.0, yaw = 0.0;
  double s, d;
  tie(s, d) = helpers.getFrenet(x, y, yaw);
  const double speed = 22.352;
  const double delta_s = 100.0, delta_d = constants.laneWidth;
  vector<vector<double>> sensorFusion;
  const VehicleData vehicleData(x, y, s, d, yaw, speed, sensorFusion);
  Path::Trajectory trajectory = Path::TrajectoryCalculator::AdjustSpeed(Path::Logic::None, vehicleData.ego, delta_s, delta_d, 0.0);
  Path::Trajectory::Kinematics kinematics = trajectory.getKinematics();
  REQUIRE(trajectory.x.front() == Approx(x).margin(0.5));
  REQUIRE(trajectory.x.back() == Approx(x + delta_s).margin(0.1)); // delta_s = delta_x in this test map
  REQUIRE(trajectory.y.front() == Approx(y).margin(0.005));
  REQUIRE(trajectory.y.back() == Approx(y - delta_d).margin(0.005)); // delta_d = -delta_y in this test map
  REQUIRE(kinematics.yaws.front() == Approx(yaw).margin(0.01));
  REQUIRE(kinematics.yaws.back() == Approx(yaw).margin(0.01));
  REQUIRE(*min_element(kinematics.speeds.begin(), kinematics.speeds.end()) == Approx(speed).margin(0.1));
  REQUIRE(*max_element(kinematics.speeds.begin(), kinematics.speeds.end()) == Approx(speed).margin(0.1));
  REQUIRE(*min_element(kinematics.accelerations.begin(), kinematics.accelerations.end()) == Approx(0.0).margin(0.1));
  REQUIRE(*max_element(kinematics.accelerations.begin(), kinematics.accelerations.end()) == Approx(0.0).margin(0.1));
  REQUIRE(*min_element(kinematics.jerks.begin(), kinematics.jerks.end()) == Approx(0.0).margin(0.1));
  REQUIRE(*max_element(kinematics.jerks.begin(), kinematics.jerks.end()) == Approx(0.0).margin(0.1));
  REQUIRE(*min_element(kinematics.yawRates.begin(), kinematics.yawRates.end()) == Approx(0.0).margin(0.06));
  REQUIRE(*max_element(kinematics.yawRates.begin(), kinematics.yawRates.end()) == Approx(0.0).margin(0.06));
}

TEST_CASE("Trajectory should smoothly accelerate from A to B", "[path]") {
  const double x = 100.0, y = 0.0, yaw = 0.0;
  double s, d;
  tie(s, d) = helpers.getFrenet(x, y, yaw);
  const double speed = 22.352, delta_speed = 5.0;
  const double delta_s = 100.0, delta_d = constants.laneWidth;
  vector<vector<double>> sensorFusion;
  const VehicleData vehicleData(x, y, s, d, yaw, speed, sensorFusion);
  Path::Trajectory trajectory = Path::TrajectoryCalculator::AdjustSpeed(Path::Logic::None, vehicleData.ego, delta_s, delta_d, delta_speed);
  Path::Trajectory::Kinematics kinematics = trajectory.getKinematics();
  REQUIRE(trajectory.x.front() == Approx(x).margin(0.5));
  REQUIRE(trajectory.x.back() == Approx(x + delta_s).margin(0.5)); // delta_s = delta_x in this test map
  REQUIRE(trajectory.y.front() == Approx(y).margin(0.005));
  REQUIRE(trajectory.y.back() == Approx(y - delta_d).margin(0.005)); // delta_d = -delta_y in this test map
  REQUIRE(kinematics.yaws.front() == Approx(yaw).margin(0.01));
  REQUIRE(kinematics.yaws.back() == Approx(yaw).margin(0.01));
  REQUIRE(kinematics.speeds.front() == Approx(speed).margin(0.05));
  REQUIRE(kinematics.speeds.back() == Approx(speed + delta_speed).margin(0.05));
  REQUIRE(*min_element(kinematics.accelerations.begin(), kinematics.accelerations.end()) == Approx(0.0).margin(10));
  REQUIRE(*max_element(kinematics.accelerations.begin(), kinematics.accelerations.end()) == Approx(0.0).margin(10));
  REQUIRE(*min_element(kinematics.jerks.begin(), kinematics.jerks.end()) == Approx(0.0).margin(0.16));
  REQUIRE(*max_element(kinematics.jerks.begin(), kinematics.jerks.end()) == Approx(0.0).margin(0.16));
  REQUIRE(*min_element(kinematics.yawRates.begin(), kinematics.yawRates.end()) == Approx(0.0).margin(0.06));
  REQUIRE(*max_element(kinematics.yawRates.begin(), kinematics.yawRates.end()) == Approx(0.0).margin(0.06));
}

TEST_CASE("Trajectory should smoothly decelerate from A to B", "[path]") {
  const double x = 100.0, y = 0.0, yaw = 0.0;
  double s, d;
  tie(s, d) = helpers.getFrenet(x, y, yaw);
  const double speed = 22.352, delta_speed = -5.0;
  const double delta_s = 100.0, delta_d = constants.laneWidth;
  vector<vector<double>> sensorFusion;
  const VehicleData vehicleData(x, y, s, d, yaw, speed, sensorFusion);
  Path::Trajectory trajectory = Path::TrajectoryCalculator::AdjustSpeed(Path::Logic::None, vehicleData.ego, delta_s, delta_d, delta_speed);
  Path::Trajectory::Kinematics kinematics = trajectory.getKinematics();
  REQUIRE(trajectory.x.front() == Approx(x).margin(0.5));
  REQUIRE(trajectory.x.back() == Approx(x + delta_s).margin(0.5)); // delta_s = delta_x in this test map
  REQUIRE(trajectory.y.front() == Approx(y).margin(0.005));
  REQUIRE(trajectory.y.back() == Approx(y - delta_d).margin(0.005)); // delta_d = -delta_y in this test map
  REQUIRE(kinematics.yaws.front() == Approx(yaw).margin(0.01));
  REQUIRE(kinematics.yaws.back() == Approx(yaw).margin(0.01));
  REQUIRE(kinematics.speeds.front() == Approx(speed).margin(0.05));
  REQUIRE(kinematics.speeds.back() == Approx(speed + delta_speed).margin(0.05));
  REQUIRE(*min_element(kinematics.accelerations.begin(), kinematics.accelerations.end()) == Approx(0.0).margin(10));
  REQUIRE(*max_element(kinematics.accelerations.begin(), kinematics.accelerations.end()) == Approx(0.0).margin(10));
  REQUIRE(*min_element(kinematics.jerks.begin(), kinematics.jerks.end()) == Approx(0.0).margin(0.15));
  REQUIRE(*max_element(kinematics.jerks.begin(), kinematics.jerks.end()) == Approx(0.0).margin(0.15));
  REQUIRE(*min_element(kinematics.yawRates.begin(), kinematics.yawRates.end()) == Approx(0.0).margin(0.06));
  REQUIRE(*max_element(kinematics.yawRates.begin(), kinematics.yawRates.end()) == Approx(0.0).margin(0.06));
}

TEST_CASE("Trajectory should smoothly continue in lane", "[path]") {
  const double x = 100.0, y = 0.0, yaw = 0.0;
  double s, d;
  tie(s, d) = helpers.getFrenet(x, y, yaw);
  const double speed = 22.352;
  const int numCoords = 100;

  vector<vector<double>> sensorFusion;
  const VehicleData vehicleData(x, y, s, d, yaw, speed, sensorFusion);
  Path::Trajectory trajectory = Path::TrajectoryCalculator::ConstantSpeed(vehicleData.ego, numCoords);
  Path::Trajectory::Kinematics kinematics = trajectory.getKinematics();
  REQUIRE(trajectory.x.front() == Approx(x).margin(0.5));
  REQUIRE(trajectory.x.back() == Approx(x + speed * constants.deltaTime * numCoords).margin(0.5));
  REQUIRE(trajectory.y.front() == Approx(y).margin(0.001));
  REQUIRE(trajectory.y.back() == Approx(y).margin(0.001));
  REQUIRE(kinematics.yaws.front() == Approx(yaw).margin(0.001));
  REQUIRE(kinematics.yaws.back() == Approx(yaw).margin(0.001));
  REQUIRE(*min_element(kinematics.speeds.begin(), kinematics.speeds.end()) == Approx(speed).margin(0.1));
  REQUIRE(*max_element(kinematics.speeds.begin(), kinematics.speeds.end()) == Approx(speed).margin(0.1));
  REQUIRE(*min_element(kinematics.accelerations.begin(), kinematics.accelerations.end()) == Approx(0.0).margin(0.1));
  REQUIRE(*max_element(kinematics.accelerations.begin(), kinematics.accelerations.end()) == Approx(0.0).margin(0.1));
  REQUIRE(*min_element(kinematics.jerks.begin(), kinematics.jerks.end()) == Approx(0.0).margin(0.1));
  REQUIRE(*max_element(kinematics.jerks.begin(), kinematics.jerks.end()) == Approx(0.0).margin(0.1));
  REQUIRE(*min_element(kinematics.yawRates.begin(), kinematics.yawRates.end()) == Approx(0.0).margin(0.06));
  REQUIRE(*max_element(kinematics.yawRates.begin(), kinematics.yawRates.end()) == Approx(0.0).margin(0.06));
}
