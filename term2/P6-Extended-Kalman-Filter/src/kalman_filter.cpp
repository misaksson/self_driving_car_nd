#include "kalman_filter.h"
#include <cmath>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Predict() {
  // Predict the new state estimation.
  x_ = F_ * x_;

  // Update the state uncertainty covariance matrix.
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {

  // Calculate the error between the measurement z and predicted state x_.
  VectorXd y = z - (H_ * x_);

  UpdateCommon(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {

  /* Calculate the error between the measurement z and predicted state x_.
   * The measurement function h(x) is the non-linear mapping between cartesian
   * and polar coordinates. */
  VectorXd y = z - Tools::CartesianToPolar(x_);

  // Normalize the angle phi to range -pi to pi.
  while (y[1] < -M_PI) {
    y[1] += 2.0 * M_PI;
  }
  while (y[1] > M_PI) {
    y[1] -= 2.0 * M_PI;
  }

  UpdateCommon(y);
}

void KalmanFilter::UpdateCommon(const VectorXd &y) {

  // Measurement matrix transpose
  MatrixXd Ht = H_.transpose();

  // Innovation covariance
  MatrixXd S = H_ * P_ * Ht + R_;

  // Kalman gain
  MatrixXd K = P_ * Ht * S.inverse();

  // Update state estimation
  x_ += K * y;

  // Update state uncertainty covariance matrix
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
