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
}
