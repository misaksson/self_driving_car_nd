#include "catch.hpp"
#include "../src/cte_eval.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <random>

using namespace std;

TEST_CASE("CTE should be evaluated as IDEAL when error is constant small", "[cte_eval]") {
  CrosstrackErrorEvaluator cteEval(false);
  double deltaTime = 0.05;
  double cte = 0.01;
  for (int i = 0; i < 100; ++i) {
    REQUIRE(cteEval.Evaluate(deltaTime, cte) == CrosstrackErrorEvaluator::IDEAL);
  }
}

TEST_CASE("CTE should be evaluated as defective when error is constant large", "[cte_eval]") {
  CrosstrackErrorEvaluator cteEval(false);
  double deltaTime = 0.05;
  double cte = 1.0;
  for (int i = 0; i < 100; ++i) {
    REQUIRE(cteEval.Evaluate(deltaTime, cte) == CrosstrackErrorEvaluator::DEFECTIVE);
  }
}

TEST_CASE("CTE should be evaluated as defective when error is small but with large deviations", "[cte_eval]") {
  CrosstrackErrorEvaluator cteEval(false);
  double deltaTime = 0.05;
  for (int i = 0; i < 100; ++i) {
    double cte = i % 2 ? 0.3 : -0.3;
    CrosstrackErrorEvaluator::Performance performance = cteEval.Evaluate(deltaTime, cte);
    if (i != 0) {
      REQUIRE(performance == CrosstrackErrorEvaluator::DEFECTIVE);
    }
  }
}

TEST_CASE("CTE should be evaluated as defective until error becomes both constant and small", "[cte_eval]") {
  CrosstrackErrorEvaluator cteEval(false);
  double deltaTime = 0.05;
  for (int i = 0; i < 100; ++i) {
    // Large constant CTE.
    double largeConstantCte = 1.0;
    if (i > 5) {
      REQUIRE(cteEval.Evaluate(deltaTime, largeConstantCte) == CrosstrackErrorEvaluator::DEFECTIVE);
    }
  }
  for (int i = 0; i < 100; ++i) {
    // Small but deviating CTE.
    double smallDeviatingCte = i % 2 ? 0.1 : -0.1;
    REQUIRE(cteEval.Evaluate(deltaTime, smallDeviatingCte) == CrosstrackErrorEvaluator::DEFECTIVE);
  }
  double smallConstantCte = 0.1;
  for (int i = 0; i < 10; ++i) {
    // Give the algorithm some time to transition.
    (void)cteEval.Evaluate(deltaTime, smallConstantCte);
  }
  REQUIRE(cteEval.Evaluate(deltaTime, smallConstantCte) == CrosstrackErrorEvaluator::IDEAL);
}
