#include "catch.hpp"
#include "../src/twiddle.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <random>

using namespace std;

TEST_CASE("Twiddle should behave just as the PID controller when inactive", "[twiddle]") {
  PID pid;
  Twiddle twiddle;
  twiddle.Init(1.0, 2.0, 3.0);
  pid.Init(1.0, 2.0, 3.0);

  default_random_engine gen;
  uniform_real_distribution<> dis(-10.0, 10.0);
  for (int i = 0; i < 1000; ++i) {
    const double cte = dis(gen);
    REQUIRE(twiddle.CalcError(cte) == Approx(pid.CalcError(cte)));
    twiddle.SetNextParams();
  }
}


class TestSystem {
private:
  double hidden_;
  double target_;
  double actual_;

  /** Reset system state. */
  void Reset() {
    hidden_ = 0.0;
    target_ = 0.0;
    actual_ = 0.0;
  }

  /** Update system state.
   * @output error
   */
  double Update(double controlSignal) {
    hidden_ += 0.1;
    target_ = sin(hidden_);
    actual_ += controlSignal;
    return actual_ - target_;
  }

public:

  /** Evaluate the PID controller performance.
   * @output accumulated error
   */
  double Run(PID &pid) {
    Reset();
    double accumulatedError = 0.0;
    double controlSignal = 0.0;
    for (int updateIdx = 0; updateIdx <= 100; ++updateIdx) {
      double error = Update(controlSignal);
      accumulatedError += pow(error, 2.0);
      controlSignal = -pid.CalcError(error);
    }
    return accumulatedError;
  }
};




TEST_CASE("Twiddle should improve PID controller performance", "[twiddle]") {
  TestSystem sys;
  Twiddle twiddle;
  twiddle.Init(0.01, 0.0001, 0.03, true, "", false);
  double lowestError = HUGE_VAL;
  array<int, 4> verifyRuns = {0, 100, 500, 1000};
  auto nextVerifyRun = verifyRuns.begin();

  for (int runIdx = 0; runIdx <= 1000; ++runIdx) {
    if (runIdx != *nextVerifyRun) {
      (void)sys.Run(twiddle);
      twiddle.SetNextParams();
    } else {
      twiddle.Abort(); // Reset best parameters
      double error = sys.Run(twiddle);
      REQUIRE(error < lowestError);
      lowestError = error;
      twiddle.Continue();
      nextVerifyRun++;
    }
  }
  twiddle.Abort();
}
