#include "catch.hpp"
#include "../src/kalman_filter.h"
#include <cmath>
#include <vector>
#include <iostream>
#include "../src/Eigen/Dense"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace std;

TEST_CASE("Kalman filter standard predict and update", "[kalman]") {
  // The input and reference result is extracted from the Lidar lesson.

  KalmanFilter ekf;
  ekf.x_ = VectorXd(4);
  ekf.P_ = MatrixXd(4, 4);
  ekf.R_ = MatrixXd(2, 2);
  ekf.F_ = MatrixXd(4, 4);
  ekf.H_ = MatrixXd(2, 4);
  ekf.Q_ = MatrixXd(4, 4);

  // Setup constants and initial states
  ekf.R_ << 0.0225, 0, 0, 0.0225;
  ekf.H_ << 1, 0, 0, 0, 0, 1, 0, 0;
  ekf.F_ << 1, 0, 0.1, 0, 0, 1, 0, 0.1, 0, 0, 1, 0, 0, 0, 0, 1;
  ekf.Q_ << 0.000125, 0, 0.0025, 0, 0, 0.000125, 0, 0.0025, 0.0025, 0, 0.05, 0, 0, 0.0025, 0, 0.05;
  ekf.x_ << 0.463227, 0.607415, 0, 0;
  ekf.P_ << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 1000;


  // Setup test vector
  typedef struct {
    VectorXd in_z;
    VectorXd expected_x;
    MatrixXd expected_P;
  } TestElem;

  VectorXd in_z(2);
  VectorXd expected_x(4);
  MatrixXd expected_P(4, 4);
  vector<TestElem> testVector;

  in_z << 0.968521, 0.40545;
  expected_x << 0.96749, 0.405862, 4.58427, -1.83232;
  expected_P << 0.0224541, 0, 0.204131, 0, 0, 0.0224541, 0, 0.204131, 0.204131, 0, 92.7797, 0, 0, 0.204131, 0, 92.7797;
  testVector.push_back({.in_z=in_z, .expected_x=expected_x, .expected_P=expected_P});

  in_z << 0.947752, 0.636824;
  expected_x << 0.958365, 0.627631, 0.110368, 2.04304;
  expected_P << 0.0220006, 0, 0.210519, 0, 0, 0.0220006, 0, 0.210519, 0.210519, 0, 4.08801, 0, 0, 0.210519, 0, 4.08801;
  testVector.push_back({.in_z=in_z, .expected_x=expected_x, .expected_P=expected_P});

  in_z << 1.42287, 0.264328;
  expected_x << 1.34291, 0.364408, 2.32002, -0.722813;
  expected_P << 0.0185328, 0, 0.109639, 0, 0, 0.0185328, 0, 0.109639, 0.109639, 0, 1.10798, 0, 0, 0.109639, 0, 1.10798;
  testVector.push_back({.in_z=in_z, .expected_x=expected_x, .expected_P=expected_P});

  // Run test
  for (auto t = testVector.begin(); t != testVector.end(); ++t)
  {
    ekf.Predict();
    ekf.Update(t->in_z);
    for (int i = 0; i < t->expected_x.size(); ++i)
    {
      REQUIRE(ekf.x_(i) == Approx(t->expected_x(i)).margin(0.00001f));
    }
    for (int i = 0; i < t->expected_P.size(); ++i)
    {
      REQUIRE(ekf.P_(i) == Approx(t->expected_P(i)));
    }
  }
}

// ToDo: Add test case for extended Kalman filter.
