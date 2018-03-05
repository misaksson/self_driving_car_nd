#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H

#include "Eigen-3.3/Eigen/Core"

class Polynomial {
public:
  /** Fit a polynomial to given points. */
  static Eigen::VectorXd Fit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order);
  /** Evaluate a polynomial. */
  static double Evaluate(Eigen::VectorXd coeffs, double x);
  /** Calculate the derivative of a polynomial. */
  static Eigen::VectorXd Derivative(Eigen::VectorXd coeffs);
};

#endif /* POLYNOMIAL_H */
