#include "catch.hpp"
#include "../src/ukf.h"
#include "../src/tools.h"
#include "../src/Eigen/Dense"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace std;

TEST_CASE("Kalman filter state initialization from lidar measurement", "[initialization]") {
  // Initialize
  VectorXd expected_x_state = VectorXd(5);
  expected_x_state << 17.43, 12.12, 0.0, 0.0, 0.0;
  MatrixXd expected_P_state = Eigen::MatrixXd::Constant(5, 5, 42.0);
  MeasurementPackage measurementPackage = {
    .timestamp_ = 42ll,
    .sensor_type_=MeasurementPackage::LASER,
    .raw_measurements_ = expected_x_state,
  };
  UKF ukf;
  REQUIRE(!ukf.is_initialized_);
  ukf.P_ = expected_P_state; // Should remain unchanged

  // Run
  ukf.ProcessMeasurement(measurementPackage);

  // Verify
  REQUIRE(ukf.is_initialized_);
  REQUIRE(ukf.previous_timestamp_ == 42ll);
  for (int i = 0; i < expected_x_state.size(); ++i)
  {
    REQUIRE(ukf.x_(i) == Approx(expected_x_state(i)));
  }
  for (int i = 0; i < expected_P_state.size(); ++i)
  {
    REQUIRE(ukf.P_(i) == Approx(expected_P_state(i)));
  }
}

TEST_CASE("Kalman filter state initialization from radar measurement", "[initialization]") {
  // Initialize
  VectorXd expected_x_state = VectorXd(5);
  expected_x_state << 17.43, 12.12, 0.0, 0.0, 0.0;
  MatrixXd expected_P_state = Eigen::MatrixXd::Constant(5, 5, 42.0);
  MeasurementPackage measurementPackage = {
    .timestamp_ = 42ll,
    .sensor_type_=MeasurementPackage::RADAR,
    .raw_measurements_ = Tools::CartesianToPolar(expected_x_state),
  };

  UKF ukf;
  REQUIRE(!ukf.is_initialized_);
  ukf.P_ = expected_P_state; // Should remain unchanged

  // Run
  ukf.ProcessMeasurement(measurementPackage);

  // Verify
  REQUIRE(ukf.is_initialized_);
  REQUIRE(ukf.previous_timestamp_ == 42ll);
  for (int i = 0; i < expected_x_state.size(); ++i)
  {
    REQUIRE(ukf.x_(i) == Approx(expected_x_state(i)));
  }
  for (int i = 0; i < expected_P_state.size(); ++i)
  {
    REQUIRE(ukf.P_(i) == Approx(expected_P_state(i)));
  }
}
