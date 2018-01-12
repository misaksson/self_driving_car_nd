#include "tools.h"
#include <cmath>
#include <iostream>

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

namespace Tools {
  VectorXd CalculateRMSE(const vector<VectorXd> &estimations,
                         const vector<VectorXd> &ground_truth) {
    /**
     * Calculates the Root Mean Squared Error (RMSE).
     */
    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;

    if ((estimations.size() == 0) ||
        (estimations.size() != ground_truth.size())) {
      cout << "CalculateRMSE: Invalid input" << endl;
    } else {
      // Accumulate squared residuals
      for (int i = 0; i < estimations.size(); ++i) {
        VectorXd error = (estimations[i] - ground_truth[i]);
        rmse = rmse.array() + error.array() * error.array();
      }

      // Calculate the mean.
      rmse /= estimations.size();

      // Calculate the squared root.
      rmse = rmse.array().sqrt();
    }
    return rmse;
  }

  MatrixXd CalculateJacobian(const VectorXd &x_state) {
    /**
     * Calculate the Jacobian matrix.
     */
    MatrixXd Hj(3, 4);
    const float px = x_state(0);
    const float py = x_state(1);
    const float vx = x_state(2);
    const float vy = x_state(3);
    const float eps = 0.00001;

    if ((fabs(px) < eps) || (fabs(py) < eps)) {
      cout << "CalculateJacobian division by zero" << endl;
    } else {
      Hj(0, 0) = px / sqrt(pow(px, 2) + pow(py, 2));
      Hj(0, 1) = py / sqrt(pow(px, 2) + pow(py, 2));
      Hj(0, 2) = 0.0;
      Hj(0, 3) = 0.0;

      Hj(1, 0) = -py / (pow(px, 2) + pow(py, 2));
      Hj(1, 1) = px / (pow(px, 2) + pow(py, 2));
      Hj(1, 2) = 0.0;
      Hj(1, 3) = 0.0;

      Hj(2, 0) = py * (vx * py - vy * px) / pow(pow(px, 2) + pow(py, 2), 1.5);
      Hj(2, 1) = px * (vy * px - vx * py) / pow(pow(px, 2) + pow(py, 2), 1.5);
      Hj(2, 2) = px / sqrt(pow(px, 2) + pow(py, 2));
      Hj(2, 3) = py / sqrt(pow(px, 2) + pow(py, 2));
    }
    return Hj;
  }

  VectorXd CartesianToPolar(const VectorXd &x) {
    const float px = x[0];
    const float py = x[1];
    const float vx = x[2];
    const float vy = x[3];

    const float rho = sqrt(pow(px, 2) + pow(py, 2));
    const float phi = atan2(py, px);
    const float rho_dot = (px * vx + py * vy) / rho;

    VectorXd result = VectorXd(3);
    result << rho, phi, rho_dot;
    return result;
  }

  VectorXd PolarToCartesian(const VectorXd &x) {
    const float rho = x[0];
    const float phi = x[1];
    const float rho_dot = x[2];

    const float px = rho * cos(phi);
    const float py = rho * sin(phi);
    const float vx = 0.0f;
    const float vy = 0.0f;

    VectorXd result = VectorXd(4);
    result << px, py, vx, vy;
    return result;
  }
}
