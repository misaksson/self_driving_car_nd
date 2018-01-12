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

  ekf_.x_ = VectorXd(4);
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.Q_ = MatrixXd(4, 4);

  // Measurement covariance matrix - laser
  R_laser_ << 0.0225f, 0.0f,
              0.0f, 0.0225f;

  // Measurement covariance matrix - radar
  R_radar_ << 0.09f, 0.0f, 0.0f,
              0.0f, 0.0009f, 0.0f,
              0.0f, 0.0f, 0.09f;

  // Laser measurement matrix
  H_laser_ << 1.0f, 0.0f, 0.0f, 0.0f,
              0.0f, 1.0f, 0.0f, 0.0f;

  // State covariance matrix
  ekf_.P_ << 1.0f, 0.0f, 0.0f, 0.0f,
             0.0f, 1.0f, 0.0f, 0.0f,
             0.0f, 0.0f, 1000.0f, 0.0f,
             0.0f, 0.0f, 0.0f, 1000.0f;
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
      ekf_.x_ = Tools::PolarToCartesian(measurement_pack.raw_measurements_);
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
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
                   1000000.0f;
  previous_timestamp_ = measurement_pack.timestamp_;
  ekf_.F_ << 1.0f, 0.0f, dt, 0.0f,
             0.0f, 1.0f, 0.0f, dt,
             0.0f, 0.0f, 1.0f, 0.0f,
             0.0f, 0.0f, 0.0f, 1.0f;

  /**
   * Update the process noise covariance matrix using
   * acceleration noise 9 in both directions.
   */
  const float noise_ax = 9.0f;
  const float noise_ay = 9.0f;
  const float dt_2 = pow(dt, 2.0f);
  const float dt_3 = 0.5f * pow(dt, 3.0f);
  const float dt_4 = 0.25f * pow(dt, 4.0f);

  ekf_.Q_ << dt_4 * noise_ax, 0.0f, dt_3 * noise_ax, 0.0f,
             0.0f, dt_4 * noise_ay, 0.0f, dt_3 * noise_ay,
             dt_3 * noise_ax, 0.0f, dt_2 * noise_ax, 0.0f,
             0.0f, dt_3 * noise_ay, 0.0f, dt_2 * noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   * Use the sensor type to perform the update step.
   * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    /* Radar
     * The non-linear measurement function is approximated by a Jacobian matrix
     * at current x_state. The extended Kalman filter is used to calculate the
     * measurement error in polar coordinates. */
    ekf_.H_ = Tools::CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    /* Laser
     * Use the H_laser measurement matrix and R_laser measurement covariance
     * matrix and do a standard Kalman filter update. */
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }
}
