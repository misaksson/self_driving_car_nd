#include "catch.hpp"
#include "../src/helper_functions.h"
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

TEST_CASE("Observations transformed to map coordinates", "[transform]") {
  /* Test data from lesson example on landmark conversion. */

  struct TestElement {
    double x;
    double y;
    double theta;
    Observation observation;

    TransformedObservation expected;
  };
  vector<TestElement> testVector;

  testVector.push_back({.x = 4.0, .y = 5.0, .theta = -M_PI / 2.0, .observation = {.x = 2.0, .y = 2.0},
                       .expected = TransformedObservation(6.0, 3.0)});
  testVector.push_back({.x = 4.0, .y = 5.0, .theta = -M_PI / 2.0, .observation = {.x = 3.0, .y = -2.0},
                       .expected = TransformedObservation(2.0, 2.0)});
  testVector.push_back({.x = 4.0, .y = 5.0, .theta = -M_PI / 2.0, .observation = {.x = 0.0, .y = -4.0},
                       .expected = TransformedObservation(0.0, 5.0)});

  for (auto testElement = testVector.begin(); testElement != testVector.end(); ++testElement) {
    const TransformedObservation actual = transformObservation(testElement->x, testElement->y, testElement->theta,
                                                               testElement->observation);
    REQUIRE(actual.x == Approx(testElement->expected.x));
    REQUIRE(actual.y == Approx(testElement->expected.y));
  }
}

TEST_CASE("Multivariate-Gaussian probability calculated correctly", "[probability]") {
  /* Test data from lesson example on particle weights. */
  struct TestElement {
    double x;
    double y;
    double mu_x;
    double mu_y;
    double sigma_x;
    double sigma_y;

    double expected;
  };
  vector<TestElement> testVector;
  testVector.push_back({.x = 6.0, .y = 3.0, .mu_x = 5.0, .mu_y = 3.0, .sigma_x = 0.3, .sigma_y = 0.3,
    .expected = 0.00683644777551});
  testVector.push_back({.x = 2.0, .y = 2.0, .mu_x = 2.0, .mu_y = 1.0, .sigma_x = 0.3, .sigma_y = 0.3,
    .expected = 0.00683644777551});
  testVector.push_back({.x = 0.0, .y = 5.0, .mu_x = 2.0, .mu_y = 1.0, .sigma_x = 0.3, .sigma_y = 0.3,
    .expected = 9.83184874151e-49});

  for (auto testElement = testVector.begin(); testElement != testVector.end(); ++testElement) {
    double actual = multivariateGaussianProbability(testElement->x, testElement->y,
                                                    testElement->mu_x, testElement->mu_y,
                                                    testElement->sigma_x, testElement->sigma_y);
    REQUIRE(actual == Approx(testElement->expected));
  }
}
