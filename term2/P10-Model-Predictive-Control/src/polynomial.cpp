#include <assert.h>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "polynomial.h"

Eigen::VectorXd Polynomial::Fit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order) {
  // The polynomial fit code was migrated to c++ by Udacity and originates from
  // https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

double Polynomial::Evaluate(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

Eigen::VectorXd Polynomial::Derivative(Eigen::VectorXd coeffs) {
  Eigen::VectorXd derivativeCoeffs(coeffs.size() - 1);
  for (int i = 1; i < coeffs.size(); ++i) {
    derivativeCoeffs[i - 1] = coeffs[i] * i;
  }
  return derivativeCoeffs;
}
