#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* state covariance matrix
  MatrixXd P_;

  ///* process noise covariance matrix
  MatrixXd Q_;

  ///* predicted sigma points matrix
  MatrixXd Xsig_pred_;

  ///* sigma point deviation from state estimate x
  MatrixXd Xsig_deviation_;

  ///* Radar measurement covariance matrix
  MatrixXd R_radar_;

  ///* Lidar measurement covariance matrix
  MatrixXd R_lidar_;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* State dimension
  int n_x_;

  ///* Augmented state dimension
  int n_aug_;

  ///* Sigma point spreading parameter
  double lambda_;

  ///* Time stamp of previous measurement
  long long previous_timestamp_;

  ///* Normalized innovation squared (NIS) value of last measurement.
  double NIS_value_;

  /**
   * Constructor
   * @param std_a Process noise standard deviation longitudinal acceleration in m/s^2
   * @param std_yawdd Process noise standard deviation yaw acceleration in rad/s^2
   */
  UKF(const double std_a = 0.46, const double std_yawdd = 0.54);

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

private:
  /** Initialize the x state estimate from the first measurement.
   * To not break the simulation, this is done even if that measurement should
   * be ignored (controlled by class members use_laser and use_radar).
   * @param meas_package The first measurement
   */
  void Initialize(const MeasurementPackage &meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);

  /**
   * Common parts of the update step.
   * This is a helper function used both for LIDAR and RADAR measurements.
   * @param Zsig_deviation Sigma point deviation from predicted measurement.
   * @param y Measurement error.
   * @param R Measurement covariance matrix.
   */
  void CommonUpdate(const MatrixXd &Zsig_deviation, const VectorXd &y, const MatrixXd &R);

};

#endif /* UKF_H */
