#include "catch.hpp"
#include "../src/Eigen/Dense"
#include "../src/FusionEKF.h"
#include "../src/measurement_package.h"
#include <cmath>
#include <vector>
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace std;

TEST_CASE("Fusion tracking with only laser measurements", "[kalman]") {
  // The input and reference result is extracted from the Lidar lesson.

  FusionEKF tracker;

  // Setup test vector
  typedef struct {
    VectorXd in_z;
    VectorXd expected_x;
  } TestElem;


  VectorXd in_z(2);
  VectorXd expected_x(4);
  vector<TestElem> testVector;

  in_z << 0.463227, 0.607415;
  expected_x << 0.463227, 0.607415, 0.0, 0.0;
  testVector.push_back({.in_z=in_z, .expected_x=expected_x});

  in_z << 0.968521, 0.40545;
  expected_x << 0.96749, 0.405862, 4.58432, -1.83234;
  testVector.push_back({.in_z=in_z, .expected_x=expected_x});

  in_z << 0.947752, 0.636824;
  expected_x << 0.958363, 0.627633, 0.109834, 2.04351;
  testVector.push_back({.in_z=in_z, .expected_x=expected_x});

  in_z << 1.42287, 0.264328;
  expected_x << 1.34304, 0.364245, 2.32719, -0.731892;
  testVector.push_back({.in_z=in_z, .expected_x=expected_x});

  MeasurementPackage measurementPackage = {
    .timestamp_ = 0ll,
    .sensor_type_=MeasurementPackage::LASER
  };

  // Run test
  for (auto t = testVector.begin(); t != testVector.end(); ++t)
  {
    measurementPackage.raw_measurements_ = t->in_z;
    measurementPackage.timestamp_ += (long long)(1000000.0f * 0.1f);
    tracker.ProcessMeasurement(measurementPackage);
    for (int i = 0; i < t->expected_x.size(); ++i)
    {
      REQUIRE(tracker.ekf_.x_(i) == Approx(t->expected_x(i)).margin(0.00001f));
    }
  }
}

// ToDo: Add test case including radar measurements.
