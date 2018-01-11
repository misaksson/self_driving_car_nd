#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

namespace Tools {

  /**
   * A helper method to calculate RMSE.
   */
  VectorXd CalculateRMSE(const vector<VectorXd> &estimations,
                         const vector<VectorXd> &ground_truth);

  /**
   * A helper method to calculate Jacobians.
   */
  MatrixXd CalculateJacobian(const VectorXd &x_state);

  /**
   * A helper method to convert cartesian to polar coordinates.
   */
  VectorXd CartesianToPolar(const VectorXd &x);

  /**
   * A helper method to convert polar to cartesian coordinates.
   */
  VectorXd PolarToCartesian(const VectorXd &x);

}

#endif /* TOOLS_H_ */
