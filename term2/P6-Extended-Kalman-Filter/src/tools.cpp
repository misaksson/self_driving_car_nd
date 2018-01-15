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

    /* Pre-calculate values used in multiple entries of the Jacobian matrix. */
    const float rho_sq = pow(px, 2) + pow(py, 2);
    const float rho = sqrt(rho_sq);
    const float rho_rho_sq = rho * rho_sq; // (px^2 + py^2)^(3/2)

    const float eps = 0.0001f;
    /* Handle division by zero.
     * If both px and py is zero, then several entries of the Jacobian matrix
     * will end up having NaN assuming the compiler is implemented according to
     * IEEE 754. In other words, there will not be a crash in the Jacobian
     * matrix calculation due to division by zero, but the NaN values in the
     * Jacobian matrix will persistently end up in x_state estimation and the
     * uncertainty covariance matrix P. To avoid this, I've chosen to just set
     * all values to zero, which effectively ignores this measurement (relying
     * on lidar and/or state prediction to move the estimate away from origin).
     *
     * Note that this scenario is highly unlikely when tracking an object since
     * it actually implies that the tracked object has collided with ego
     * vehicle.
     *
     * If px and py are close to zero, then (px^2 + py^2)^(3/2) will be the smallest
     * denominator so thats the only value being checked below.
     */
    if (rho_rho_sq < eps) {
      cout << "CalculateJacobian division by zero" << endl;
      Hj << 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0;

    } else {

      Hj(0, 0) = px / rho;
      Hj(0, 1) = py / rho;
      Hj(0, 2) = 0.0;
      Hj(0, 3) = 0.0;

      Hj(1, 0) = -py / rho_sq;
      Hj(1, 1) = px / rho_sq;
      Hj(1, 2) = 0.0;
      Hj(1, 3) = 0.0;

      Hj(2, 0) = py * (vx * py - vy * px) / rho_rho_sq;
      Hj(2, 1) = px * (vy * px - vx * py) / rho_rho_sq;
      Hj(2, 2) = Hj(0, 0);
      Hj(2, 3) = Hj(0, 1);
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
    const float eps = 0.0001f;

    /* Set rho_dot to zero instead of NaN when rho is zero. This to avoid
     * getting a persistent NaN state value. */
    const float rho_dot = (rho > eps) ? (px * vx + py * vy) / rho : 0.0f;

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
