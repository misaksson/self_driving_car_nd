#include "catch.hpp"
#include "../src/polynomial.h"
#include <iostream>
#include <cmath>
#include <random>
#include <vector>

using namespace std;

TEST_CASE("Polynomial should fit points", "[polynomial]") {
  Eigen::VectorXd xvals(5), yvals(5);
  xvals << 0.0, 1.0, 2.0, 3.0, 4.0;
  yvals << -10.0, 20.0, 30.0, 20.0, 15.0;

  Eigen::VectorXd coeffs = Polynomial::Fit(xvals, yvals, 3);
  for (int i = 0; i < xvals.size(); ++i) {
    double eval = Polynomial::Evaluate(coeffs, xvals[i]);
    REQUIRE(eval == Approx(yvals[i]).margin(2.5));
  }
}

TEST_CASE("Polynomial should be continuous", "[polynomial]") {
  Eigen::VectorXd xvals(5), yvals(5);
  xvals << 0.0, 1.0, 2.0, 3.0, 4.0;
  yvals << -10.0, 20.0, 30.0, 20.0, 15.0;

  Eigen::VectorXd coeffs = Polynomial::Fit(xvals, yvals, 3);

  double previous = Polynomial::Evaluate(coeffs, 0.0);
  for (double x = 0.001; x < 5.0; x += 0.001) {
    double current = Polynomial::Evaluate(coeffs, x);
    REQUIRE(current == Approx(previous).margin(0.1));
    previous = current;
  }
}

TEST_CASE("Polynomial calculates derivatives properly", "[polynomial]") {
  /* d/dx(2 x^3 + 3 x^2 - 4 x + 5) = 6 x^2 + 6 x - 4 */
  Eigen::VectorXd coeffs(4), expected(3);
  coeffs << 5.0, -4.0, 3.0, 2.0;
  expected << -4.0, 6.0, 6.0;

  Eigen::VectorXd actual = Polynomial::Derivative(coeffs);

  REQUIRE(actual.size() == expected.size());
  for (int i = 0; i < expected.size(); ++i) {
    REQUIRE(actual[i] == Approx(expected[i]));
  }
}
