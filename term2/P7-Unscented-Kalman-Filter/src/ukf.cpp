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
 */
UKF::UKF(const double std_a, const double std_yawdd) {
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

  //create matrix with predicted sigma points as columns
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

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

  // initial state vector and covariance matrix.
  x_ = VectorXd::Zero(n_x_);
  P_ = MatrixXd::Identity(n_x_, n_x_);
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
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    x_.head(2) = Tools::PolarToCartesian(meas_package.raw_measurements_);

    /**
     * TODO: calculate the covariance of px, py given an radar measurement.
     * For now, lets just assume it's two times the uncertainty of a LIDAR
     * measurement. The actual covariance should however depend on both the
     * measured value of rho and phi.
     */
    P_.topLeftCorner(2, 2) = R_lidar_ * 2.0;
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    x_.head(2) = meas_package.raw_measurements_;

    // Use same covariance as in the measurement.
    P_.topLeftCorner(2, 2) = R_lidar_;
  }

  /**
   * None of the sensors provides direct information about v, yaw and yawd.
   * Radar measurements does however contain rho_dot, but without the angular
   * velocity phi_dot, this is not easily transformable to the CTRV motion
   * model and using that information for the initial estimate is considered
   * beyond the scope of this project.
   *
   * One might assume that a bicycle in average will go 5 m/s. Attempts to use
   * that assumption did however cause worse RMSE when the initial yaw angle
   * happens to be in the opposite direction (data set 2). Setting the initial
   * yaw angle covariance to a huge value didn't prevent this.
   */
  x_.tail(3) << 0.0, 0.0, 0.0;
  P_(2, 2) = 5.0;
  P_(3, 3) = 2.0 * M_PI;
  P_(4, 4) = 0.5 * M_PI;

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
  for (int i = 0; i < n_aug_; ++i) {
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


  // Predict state mean.
  x_ = Xsig_pred_ * weights_;

  // Calculate sigma point deviation from state mean estimate x.
  Xsig_deviation_ = Xsig_pred_.colwise() - x_;
  Tools::NormalizeAngles(Xsig_deviation_, 3); // Normalize yaw angles at row 3

  // Predict state covariance matrix.
  P_ = (Xsig_deviation_.array().rowwise() * weights_.transpose().array()).matrix() * Xsig_deviation_.transpose();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  const int n_z = 2;
  const VectorXd z = meas_package.raw_measurements_;
  assert(z.size() == n_z);

  // Transform sigma points into measurement space.
  const MatrixXd Zsig = Xsig_pred_.topRows(n_z);

  // Calculate mean predicted measurement.
  const VectorXd z_pred = Zsig * weights_;

  // Calculate sigma points deviation from mean predicted measurement.
  const MatrixXd Zsig_deviation = Zsig.colwise() - z_pred;

  // Calculate measurement error y.
  const VectorXd y = z - z_pred;

  CommonUpdate(Zsig_deviation, y, R_lidar_);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  const int n_z = 3;
  const VectorXd z = meas_package.raw_measurements_;
  assert(z.size() == n_z);

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

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

  // Calculate mean predicted measurement.
  const VectorXd z_pred = Zsig * weights_;

  // Calculate sigma points deviation from mean predicted measurement.
  MatrixXd Zsig_deviation = Zsig.colwise() - z_pred;
  Tools::NormalizeAngles(Zsig_deviation, 1);  // Normalize phi angles at row 1

  // Calculate measurement error y.
  VectorXd y = z - z_pred;
  Tools::NormalizeAngles(y(1));  // Normalize phi angle

  CommonUpdate(Zsig_deviation, y, R_radar_);
}

void UKF::CommonUpdate(const MatrixXd &Zsig_deviation, const VectorXd &y, const MatrixXd &R) {
  // Calculate innovation covariance matrix S
  const MatrixXd S = (Zsig_deviation.array().rowwise() * weights_.transpose().array()).matrix() *
                      Zsig_deviation.transpose() + R;
  const MatrixXd S_inv = S.inverse();

  // Calculate cross correlation matrix Tc
  const MatrixXd Tc = (Xsig_deviation_.array().rowwise() * weights_.transpose().array()).matrix() *
                       Zsig_deviation.transpose();

  // Calculate Kalman gain K
  const MatrixXd K = Tc * S_inv;

  // Update state mean and covariance matrix.
  x_ += K * y;
  P_ -= K * S * K.transpose();

  // Calculate the normalized innovation squared (NIS) value
  NIS_value_ = y.transpose() * S_inv * y;
}
