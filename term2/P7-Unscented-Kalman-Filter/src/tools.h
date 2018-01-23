#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

namespace Tools {

  /**
  * A helper method to calculate RMSE.
  */
  VectorXd CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);

  /**
   * A helper method to convert cartesian to polar coordinates.
   */
  VectorXd CartesianToPolar(const VectorXd &x);

  /**
   * A helper method to convert polar to cartesian coordinates.
   */
  VectorXd PolarToCartesian(const VectorXd &x);

  /**
   * Normalize angles to range [-PI, PI].
   */
  void NormalizeAngles(double &angle);

  /**
   * Normalize angles at matrix row to range [-PI, PI].
   */
  void NormalizeAngles(MatrixXd &angles, int row);
};

#endif /* TOOLS_H_ */
