#include "ukf.h"
#include "Eigen/Dense"
#include "tools.h"
#include <iostream>
#include <cmath>
#include <assert.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // Use first measurement for initialization.
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = (double)(3 - n_aug_);

  // Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  weights_.tail(weights_.size() - 1) = VectorXd::Constant(weights_.size() - 1, 0.5 / (lambda_ + n_aug_));

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  // ToDo: Let this reflect the uncertainty of the first measurement.
  P_ = MatrixXd(n_x_, n_x_);
  P_ << 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0;

  //create matrix with predicted sigma points as columns
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  const double std_a = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  const double std_yawdd = 0.5;

  // process noise covariance matrix
  Q_ = MatrixXd(2, 2);
  Q_ << pow(std_a, 2.0), 0.0,
        0.0, pow(std_yawdd, 2.0);

  // Laser measurement noise standard deviation for x dimension in m
  const double std_laspx = 0.15;

  // Laser measurement noise standard deviation for y dimension in m
  const double std_laspy = 0.15;

  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << pow(std_laspx, 2.0), 0.0,
              0.0, pow(std_laspy, 2.0);

  // Radar measurement noise standard deviation radius in m
  const double std_radr = 0.3;

  // Radar measurement noise standard deviation angle in rad
  const double std_radphi = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  const double std_radrd = 0.3;

  // Radar measurement covariance matrix
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << pow(std_radr, 2.0), 0.0, 0.0,
              0.0, pow(std_radphi, 2.0), 0.0,
              0.0, 0.0, pow(std_radrd, 2.0);

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  if (!is_initialized_) {
    Initialize(meas_package);
    return; // Only use first measurement to initialize.
  }


  if ((use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) ||
      (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER)) {

    // Calculate delta time in seconds
    const double delta_t = (double)(meas_package.timestamp_ - previous_timestamp_) /
                           1000000.0;
    previous_timestamp_ = meas_package.timestamp_;
    Prediction(delta_t);
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      UpdateRadar(meas_package);
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      UpdateLidar(meas_package);
    }
  }
}

void UKF::Initialize(const MeasurementPackage &meas_package) {
  double px, py;
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    const VectorXd cartesian = Tools::PolarToCartesian(meas_package.raw_measurements_);
    px = cartesian(0);
    py = cartesian(1);
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    px = meas_package.raw_measurements_[0];
    py = meas_package.raw_measurements_[1];
  }
  const double v = 0.0f;
  const double yaw = 0.0f;
  const double yawd = 0.0f;
  x_ << px, py, v, yaw, yawd;

  previous_timestamp_ = meas_package.timestamp_;
  is_initialized_ = true;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  //create augmented mean vector
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  x_aug.head(n_x_) = x_;

  //create augmented covariance matrix
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(n_aug_ - n_x_, n_aug_ - n_x_) = Q_;

  //create square root matrix
  MatrixXd A_aug = P_aug.llt().matrixL();

  //create augmented sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.col(0) = x_aug;
  const double lambdaFactor = sqrt(lambda_ + (double)n_aug_);
  for (int i = 0; i < n_aug_; ++i)
  {
    Xsig_aug.col(i + 1) = x_aug + lambdaFactor * A_aug.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - lambdaFactor * A_aug.col(i);
  }

  //predict sigma points
  for (int i = 0; i < 2* n_aug_ + 1; ++i) {
    // Extract sigma point values at time k
    const double px = Xsig_aug(0, i);
    const double py = Xsig_aug(1, i);
    const double v = Xsig_aug(2, i);
    const double yaw = Xsig_aug(3, i);
    const double yawd = Xsig_aug(4, i);
    const double nu_a = Xsig_aug(5, i);
    const double nu_yawdd = Xsig_aug(6, i);

    // Predict sigma point values at time k + delta_t
    double px_p, py_p;
    const double eps = 0.0001;
    // Avoid division by zero
    if (fabs(yawd) < eps) {
      px_p = px + (v * cos(yaw) * delta_t) + (0.5 * pow(delta_t, 2.0) * cos(yaw) * nu_a);
      py_p = py + (v * sin(yaw) * delta_t) + (0.5 * pow(delta_t, 2.0) * sin(yaw) * nu_a);
    } else {
      px_p = px + ((v / yawd) * (sin(yaw + yawd * delta_t) - sin(yaw))) + (0.5 * pow(delta_t, 2.0) * cos(yaw) * nu_a);
      py_p = py + ((v / yawd) * (-cos(yaw + yawd * delta_t) + cos(yaw))) + (0.5 * pow(delta_t, 2.0) * sin(yaw) * nu_a);
    }
    const double v_p = v + (delta_t * nu_a);
    const double yaw_p = yaw + (yawd * delta_t) + (0.5 * pow(delta_t, 2) * nu_yawdd);
    const double yawd_p = yawd + (delta_t * nu_yawdd);

    Xsig_pred_.col(i) << px_p, py_p, v_p, yaw_p, yawd_p;
  }

  //predict state mean
  x_ = Xsig_pred_ * weights_;

  //predict state covariance matrix
  MatrixXd xDiff = Xsig_pred_.colwise() - x_;
  MatrixXd::Index idx;
  while (xDiff.row(3).minCoeff(&idx) < M_PI) {
    xDiff(3, idx) += 2.0 * M_PI;
  }
  while (xDiff.row(3).maxCoeff(&idx) > M_PI) {
    xDiff(3, idx) -= 2.0 * M_PI;
  }
  P_ = (xDiff.array().rowwise() * weights_.transpose().array()).matrix() * xDiff.transpose();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO: You'll also need to calculate the lidar NIS.
  */
  const int n_z = 2;
  const VectorXd z = meas_package.raw_measurements_;
  assert(z.size() == n_z);

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  //transform sigma points into measurement space
  Zsig = Xsig_pred_.topRows(n_z);

  //calculate mean predicted measurement
  z_pred = Zsig * weights_;

  //calculate innovation covariance matrix S
  MatrixXd zDiff = Zsig.colwise() - z_pred;
  S = (zDiff.array().rowwise() * weights_.transpose().array()).matrix() * zDiff.transpose() + R_lidar_;

  //calculate cross correlation matrix Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  MatrixXd xDiff = Xsig_pred_.colwise() - x_;
  MatrixXd::Index idx;
  while (xDiff.row(3).minCoeff(&idx) < M_PI) {
    xDiff(3, idx) += 2.0 * M_PI;
  }
  while (xDiff.row(3).maxCoeff(&idx) > M_PI) {
    xDiff(3, idx) -= 2.0 * M_PI;
  }
  Tc = (xDiff.array().rowwise() * weights_.transpose().array()).matrix() * zDiff.transpose();

  //calculate Kalman gain K
  MatrixXd K = Tc * S.inverse();

  //update state mean and covariance matrix
  VectorXd y = z - z_pred;
  x_ += K * y;
  P_ -= K * S * K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO: You'll also need to calculate the radar NIS.
  */

  const int n_z = 3;
  const VectorXd z = meas_package.raw_measurements_;
  assert(z.size() == n_z);

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  //transform sigma points into measurement space
  for (int i = 0; i < Xsig_pred_.cols(); ++i) {
    // Extract sigma point values
    const double px = Xsig_pred_(0, i);
    const double py = Xsig_pred_(1, i);
    const double v = Xsig_pred_(2, i);
    const double yaw = Xsig_pred_(3, i);

    // Transform to radar measurement space
    const double rho = sqrt(pow(px, 2.0) + pow(py, 2.0));
    const double phi = atan2(py, px);
    const double rho_dot = (px * cos(yaw) + py * sin(yaw)) * v / rho;

    Zsig.col(i) << rho, phi, rho_dot;
  }

  //calculate mean predicted measurement
  z_pred = Zsig * weights_;

  //calculate innovation covariance matrix S
  MatrixXd zDiff = Zsig.colwise() - z_pred;
  MatrixXd::Index idx;
  while (zDiff.row(1).minCoeff(&idx) < M_PI) {
    zDiff(1, idx) += 2.0 * M_PI;
  }
  while (zDiff.row(1).maxCoeff(&idx) > M_PI) {
    zDiff(1, idx) -= 2.0 * M_PI;
  }
  S = (zDiff.array().rowwise() * weights_.transpose().array()).matrix() * zDiff.transpose() + R_radar_;

  //calculate cross correlation matrix Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  MatrixXd xDiff = Xsig_pred_.colwise() - x_;
  while (xDiff.row(3).minCoeff(&idx) < M_PI) {
    xDiff(3, idx) += 2.0 * M_PI;
  }
  while (xDiff.row(3).maxCoeff(&idx) > M_PI) {
    xDiff(3, idx) -= 2.0 * M_PI;
  }
  Tc = (xDiff.array().rowwise() * weights_.transpose().array()).matrix() * zDiff.transpose();

  //calculate Kalman gain K
  MatrixXd K = Tc * S.inverse();

  //update state mean and covariance matrix
  VectorXd y = z - z_pred;
  while (y(1) < M_PI) {
    y(1) += 2.0 * M_PI;
  }
  while (y(1) > M_PI) {
    y(1) -= 2.0 * M_PI;
  }
  x_ += K * y;
  P_ -= K * S * K.transpose();
}
