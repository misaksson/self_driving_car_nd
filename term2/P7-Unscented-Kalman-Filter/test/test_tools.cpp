#include "catch.hpp"
#include "../src/tools.h"
#include <cmath>
#include "../src/Eigen/Dense"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace std;

TEST_CASE("Calculate RMSE should equal reference", "[rmse]") {
  // Test input and reference result is from the RMSE lesson example.
  vector<VectorXd> estimations;
  vector<VectorXd> ground_truth;

  //the input list of estimations
  VectorXd e(4);
  e << 1.0f, 1.0f, 0.2f, 0.1f;
  estimations.push_back(e);
  e << 2.0f, 2.0f, 0.3f, 0.2f;
  estimations.push_back(e);
  e << 3.0f, 3.0f, 0.4f, 0.3f;
  estimations.push_back(e);

  //the corresponding list of ground truth values
  VectorXd g(4);
  g << 1.1f, 1.1f, 0.3f, 0.2f;
  ground_truth.push_back(g);
  g << 2.1f, 2.1f, 0.4f, 0.3f;
  ground_truth.push_back(g);
  g << 3.1f, 3.1f, 0.5f, 0.4f;
  ground_truth.push_back(g);

  VectorXd rmse = Tools::CalculateRMSE(estimations, ground_truth);

  for (int i = 0; i < rmse.size(); ++i)
  {
    REQUIRE(rmse(i) == Approx(0.1f));
  }
}

TEST_CASE("Calculate RMSE should handle empty input", "[rmse]") {
  vector<VectorXd> estimations;
  vector<VectorXd> ground_truth;
  VectorXd rmse = Tools::CalculateRMSE(estimations, ground_truth);
  for (int i = 0; i < rmse.size(); ++i)
  {
    REQUIRE(rmse(i) == Approx(0.0f));
  }
}
