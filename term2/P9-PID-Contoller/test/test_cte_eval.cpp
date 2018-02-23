#include "catch.hpp"
#include "../src/cte_eval.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <random>

using namespace std;

TEST_CASE("CTE should be evaluated as OK when error is constant small", "[cte_eval]") {
  CrosstrackErrorEvaluator cteEval;
  double deltaTime = 0.05;
  double cte = 0.4;
  for (int i = 0; i < 1000; ++i) {
    REQUIRE(cteEval.IsOk(deltaTime, cte));
  }
}

TEST_CASE("CTE should be evaluated as NOK when error is constant large", "[cte_eval]") {
  CrosstrackErrorEvaluator cteEval;
  double deltaTime = 0.05;
  double cte = 1.0;
  for (int i = 0; i < 1000; ++i) {
    REQUIRE(!cteEval.IsOk(deltaTime, cte));
  }
}

TEST_CASE("CTE should be evaluated as NOK when error is small but with large deviations", "[cte_eval]") {
  CrosstrackErrorEvaluator cteEval;
  double deltaTime = 0.05;
  for (int i = 0; i < 100; ++i) {
    double cte = i % 2 ? 0.3 : -0.3;
    bool isOk = cteEval.IsOk(deltaTime, cte);
    if (i != 0) {
      REQUIRE(!isOk);
    }
  }
}

TEST_CASE("CTE should be evaluated as NOK until error becomes both constant and small", "[cte_eval]") {
  CrosstrackErrorEvaluator cteEval;
  double deltaTime = 0.05;
  for (int i = 0; i < 100; ++i) {
    // Large constant CTE.
    double largeConstantCte = 1.0;
    REQUIRE(!cteEval.IsOk(deltaTime, largeConstantCte));
  }
  for (int i = 0; i < 100; ++i) {
    // Small but deviating CTE.
    double smallDeviatingCte = i % 2 ? 0.1 : -0.1;
    REQUIRE(!cteEval.IsOk(deltaTime, smallDeviatingCte));
  }
  double smallConstantCte = 0.1;
  for (int i = 0; i < 10; ++i) {
    // Give the algorithm some time to transition.
    (void)cteEval.IsOk(deltaTime, smallConstantCte);
  }
  REQUIRE(cteEval.IsOk(deltaTime, smallConstantCte));
}
