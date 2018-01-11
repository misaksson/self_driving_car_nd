#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  ekf_.x_ = VectorXd(4);
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.Q_ = MatrixXd(4, 4);

  // Measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // Measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  // Laser measurement matrix
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  // State covariance matrix
  ekf_.P_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1000, 0,
             0, 0, 0, 1000;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
     * Initialize the state ekf_.x_ with the first measurement.
     */
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
       * Convert radar from polar to cartesian coordinates and initialize
       * state.
       */
      ekf_.x_ = Tools::PolarToCartesian(measurement_pack.raw_measurements_);
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
       * Lidar measurement does not give speed.
       */
      const float px = measurement_pack.raw_measurements_[0];
      const float py = measurement_pack.raw_measurements_[1];
      const float vx = 0.0f;
      const float vy = 0.0f;
      ekf_.x_ << px, py, vx, vy;
    }

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   * Update the state transition matrix F according to the new elapsed time.
   * Time is measured in seconds.
   */
  const float dt = (measurement_pack.timestamp_ - previous_timestamp_) /
                   1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;
  ekf_.F_ << 1, 0, dt, 0,
             0, 1, 0, dt,
             0, 0, 1, 0,
             0, 0, 0, 1;

  /**
   * Update the process noise covariance matrix using
   * acceleration noise 9 in both directions.
   */
  const float noise_ax = 9;
  const float noise_ay = 9;
  ekf_.Q_ << 0.25 * pow(dt, 4) * noise_ax, 0, 0.5 * pow(dt, 3) * noise_ax, 0, 0,
             0.25 * pow(dt, 4) * noise_ay, 0, 0.5 * pow(dt, 3) * noise_ay,
             0.5 * pow(dt, 3) * noise_ax, 0, pow(dt, 2) * noise_ax, 0, 0,
             0.5 * pow(dt, 3) * noise_ay, 0, pow(dt, 2) * noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   * Use the sensor type to perform the update step.
   * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }
}
