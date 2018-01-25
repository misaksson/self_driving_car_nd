#include "catch.hpp"
#include "../src/Eigen/Dense"
#include "../src/tools.h"
#include "../src/ukf.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace std;

TEST_CASE("Kalman filter state initialization from lidar measurement", "[ukf_init]") {
  // Initialize
  VectorXd expected_x_state = VectorXd(5);
  expected_x_state << 17.43, 12.12, 0.0, 0.0, 0.0;
  MeasurementPackage measurementPackage = {
    .timestamp_ = 42ll,
    .sensor_type_ = MeasurementPackage::LASER,
    .raw_measurements_ = expected_x_state.head(2),
  };
  UKF ukf;
  REQUIRE(!ukf.is_initialized_);

  // Run
  ukf.ProcessMeasurement(measurementPackage);

  // Verify
  REQUIRE(ukf.is_initialized_);
  REQUIRE(ukf.previous_timestamp_ == 42ll);
  for (int i = 0; i < expected_x_state.size(); ++i)
  {
    REQUIRE(ukf.x_(i) == Approx(expected_x_state(i)));
  }
}

TEST_CASE("Kalman filter state initialization from radar measurement", "[ukf_init]") {
  // Initialize
  VectorXd expected_x_state = VectorXd(5);
  expected_x_state << 17.43, 12.12, 0.0, 0.0, 0.0;
  MeasurementPackage measurementPackage = {
    .timestamp_ = 42ll,
    .sensor_type_ = MeasurementPackage::RADAR,
    .raw_measurements_ = Tools::CartesianToPolar(expected_x_state),
  };

  UKF ukf;
  REQUIRE(!ukf.is_initialized_);

  // Run
  ukf.ProcessMeasurement(measurementPackage);

  // Verify
  REQUIRE(ukf.is_initialized_);
  REQUIRE(ukf.previous_timestamp_ == 42ll);
  for (int i = 0; i < expected_x_state.size(); ++i)
  {
    REQUIRE(ukf.x_(i) == Approx(expected_x_state(i)));
  }
}

TEST_CASE("Kalman filter predict", "[ukf_predict]") {
  const double delta_t = 0.1; //time diff in sec
  MeasurementPackage measurementPackage = {
    .timestamp_ = 0ll,
    .sensor_type_ = MeasurementPackage::LASER,
    .raw_measurements_ = VectorXd::Zero(2),
  };
  UKF ukf;
  ukf.use_laser_ = true;
  // A first (dummy) run to initialize state
  ukf.ProcessMeasurement(measurementPackage);

  // Overwrite state by values from lesson example.
  ukf.x_ << 5.7441, 1.3800, 2.2049, 0.5015, 0.3528;
  ukf.P_ << 0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
           -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
            0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
           -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
           -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;
  ukf.Q_ << pow(0.2, 2.0), 0.0,
            0.0, pow(0.2, 2.0);

  // Force measurement uncertainty extremely high to neglect measurement update.
  ukf.R_lidar_ << 1.0E30, 0.0,
                  0.0, 1.0E30;
  measurementPackage.timestamp_ += (long long)(delta_t * 1000000.0);

  // Second run doing the state prediction.
  ukf.ProcessMeasurement(measurementPackage);

  // Verify state with expected result from lesson examples.
  MatrixXd expected_Xsig_pred = MatrixXd(5, 15);
  expected_Xsig_pred <<
    5.93553, 6.06251, 5.92217, 5.9415, 5.92361, 5.93516, 5.93705, 5.93553, 5.80832, 5.94481, 5.92935, 5.94553, 5.93589, 5.93401, 5.93553,
    1.48939, 1.44673, 1.66484, 1.49719, 1.508, 1.49001, 1.49022, 1.48939, 1.5308, 1.31287, 1.48182, 1.46967, 1.48876, 1.48855, 1.48939,
    2.2049, 2.28414, 2.24557, 2.29582, 2.2049, 2.2049, 2.23954, 2.2049, 2.12566, 2.16423, 2.11398, 2.2049, 2.2049, 2.17026, 2.2049,
    0.53678, 0.473387, 0.678098, 0.554557, 0.643644, 0.543372, 0.53678, 0.538512, 0.600173, 0.395462, 0.519003, 0.429916, 0.530188, 0.53678, 0.535048,
    0.3528, 0.299973, 0.462123, 0.376339, 0.48417, 0.418721, 0.3528, 0.387441, 0.405627, 0.243477, 0.329261, 0.22143, 0.286879, 0.3528, 0.318159;
  for (int i = 0; i < expected_Xsig_pred.size(); ++i) {
    REQUIRE(ukf.Xsig_pred_(i) == Approx(expected_Xsig_pred(i)));
  }

  VectorXd expected_x_state = VectorXd(5);
  expected_x_state << 5.93637, 1.49035, 2.20528, 0.536853, 0.353577;
  for (int i = 0; i < expected_x_state.size(); ++i) {
    REQUIRE(ukf.x_(i) == Approx(expected_x_state(i)).margin(0.002));
  }

  MatrixXd expected_P_state = MatrixXd(5, 5);
  expected_P_state << 0.00543425, -0.0024053, 0.00341576, -0.00348196, -0.00299378,
                      -0.0024053, 0.010845, 0.0014923, 0.00980182, 0.00791091,
                      0.00341576, 0.0014923, 0.00580129, 0.000778632, 0.000792973,
                      -0.00348196, 0.00980182, 0.000778632, 0.0119238, 0.0112491,
                      -0.00299378, 0.00791091, 0.000792973, 0.0112491, 0.0126972;
  for (int i = 0; i < expected_P_state.size(); ++i) {
    REQUIRE(ukf.P_(i) == Approx(expected_P_state(i)).margin(0.0003));
  }
}

TEST_CASE("Kalman filter predict and update using radar measurement", "[ukf_update]") {
  const double delta_t = 0.1; //time diff in sec
  MeasurementPackage measurementPackage = {
    .timestamp_ = 0ll,
    .sensor_type_ = MeasurementPackage::RADAR,
    .raw_measurements_ = VectorXd::Zero(3),
  };
  UKF ukf;
  ukf.use_radar_ = true;
  // A first (dummy) run to initialize state
  ukf.ProcessMeasurement(measurementPackage);

  // Overwrite state by values from lesson example.
  ukf.x_ << 5.7441, 1.3800, 2.2049, 0.5015, 0.3528;
  ukf.P_ << 0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
           -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
            0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
           -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
           -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;
  ukf.Q_ << pow(0.2, 2.0), 0.0,
            0.0, pow(0.2, 2.0);
  ukf.R_radar_ << pow(0.3, 2.0), 0.0, 0.0,
                  0.0, pow(0.0175, 2.0), 0.0,
                  0.0, 0.0, pow(0.1, 2.0);
  measurementPackage.timestamp_ += (long long)(delta_t * 1000000.0);
  measurementPackage.raw_measurements_ << 5.9214, 0.2187, 2.0062;

  // Second run doing the state prediction and update.
  ukf.ProcessMeasurement(measurementPackage);

  // Verify state with expected result from lesson examples.
  VectorXd expected_x_state = VectorXd(5);
  expected_x_state <<  5.92276, 1.41823, 2.15593, 0.489274, 0.321338;
  for (int i = 0; i < expected_x_state.size(); ++i) {
    REQUIRE(ukf.x_(i) == Approx(expected_x_state(i)).margin(0.002));
  }

  MatrixXd expected_P_state = MatrixXd(5, 5);
  expected_P_state <<  0.00361579, -0.000357881, 0.00208316, -0.000937196, -0.00071727,
                       -0.000357881, 0.00539867, 0.00156846, 0.00455342, 0.00358885,
                       0.00208316, 0.00156846, 0.00410651, 0.00160333, 0.00171811,
                       -0.000937196, 0.00455342, 0.00160333, 0.00652634, 0.00669436,
                       -0.00071719, 0.00358884, 0.00171811, 0.00669426, 0.00881797;
  for (int i = 0; i < expected_P_state.size(); ++i) {
    REQUIRE(ukf.P_(i) == Approx(expected_P_state(i)).margin(0.0003));
  }
}

