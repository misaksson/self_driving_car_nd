#include <iostream>
#include "tools.h"

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
