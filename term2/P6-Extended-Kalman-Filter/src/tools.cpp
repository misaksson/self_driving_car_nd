#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * Calculates the Root Mean Squared Error (RMSE).
   */
   VectorXd rmse(4);
   rmse << 0,0,0,0;

   if ((estimations.size() == 0) || (estimations.size() != ground_truth.size()))
   {
      cout << "CalculateRMSE: Invalid input" << endl;
   }
   else
   {
      // Accumulate squared residuals
      for(int i = 0; i < estimations.size(); ++i){
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

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */
}
