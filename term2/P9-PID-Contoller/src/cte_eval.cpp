#include <iostream>
#include <cmath>
#include <algorithm>
#include <deque>

#include "ansi_color_codes.h"
#include "cte_eval.h"

using namespace std;

CrosstrackErrorEvaluator::CrosstrackErrorEvaluator(bool consoleOutput) : consoleOutput_(consoleOutput),
                                                                         currentPerformance_(DEFECTIVE),
                                                                         previousCte_(0.0) {
  longestIntegralPeriod_ = 0.0;
  for (auto pl = performanceLevels_.begin(); pl != performanceLevels_.end(); ++pl) {
    longestIntegralPeriod_ = max(longestIntegralPeriod_, pl->triggerPeriod_);
    longestIntegralPeriod_ = max(longestIntegralPeriod_, pl->releasePeriod_);
  }
};

CrosstrackErrorEvaluator::~CrosstrackErrorEvaluator() {};

CrosstrackErrorEvaluator::Performance CrosstrackErrorEvaluator::Evaluate(double deltaTime, double cte) {
  const Performance previousPerformance = currentPerformance_;
  history_.push_front({.deltaTime = deltaTime,
                       .cteSq = pow(cte, 2.0),
                       .deltaCteSq = pow(cte - previousCte_, 2.0)});
  previousCte_ = cte;

  // Check if performance is worse.
  for (int p = currentPerformance_ - 1; p >= DEFECTIVE; --p) {
    if (TryTrigger(performanceLevels_[p])) {
      currentPerformance_ = static_cast<Performance>(p);
    } else {
      // No need to evaluate any worse level since this one didn't trigger.
      break;
    }
  }

  // Check if performance is better.
  for (int p = currentPerformance_; p <= IDEAL; ++p) {
    if (TryRelease(performanceLevels_[p])) {
      currentPerformance_ = static_cast<Performance>(p + 1);
    } else {
      // No need to evaluate any better level since this one didn't release.
      break;
    }
  }
  ReduceHistory();

  if (consoleOutput_ && previousPerformance != currentPerformance_) {
    const array<string, NUM_PERFORMANCE_LEVELS> colorMap = {
      {ANSI::RED, ANSI::PURPLE, ANSI::YELLOW, ANSI::CYAN, ANSI::BLUE, ANSI::GREEN}
    };
    cout << colorMap[currentPerformance_];
    cout << "New performance level " << performanceLevels_[currentPerformance_].name << endl;
    cout << ANSI::RESET;
  }
  return currentPerformance_;
}

bool CrosstrackErrorEvaluator::TryTrigger(const PerformanceLevel &pl) {
  double integralCteSqPerSecond, integralDeltaCteSqPerSecond;
  tie(integralCteSqPerSecond, integralDeltaCteSqPerSecond) = Integrate(pl.triggerPeriod_);

  // Only one threshold must be passed to trigger performance level.
  return (integralCteSqPerSecond > pl.cteSqUpperTh_) ||
         (integralDeltaCteSqPerSecond > pl.deltaCteSqUpperTh_);
}

bool CrosstrackErrorEvaluator::TryRelease(const PerformanceLevel &pl) {
  double integralCteSqPerSecond, integralDeltaCteSqPerSecond;
  tie(integralCteSqPerSecond, integralDeltaCteSqPerSecond) = Integrate(pl.releasePeriod_);
  // Both thresholds must be passed to release performance level.
  return (integralCteSqPerSecond < pl.cteSqLowerTh_) &&
         (integralDeltaCteSqPerSecond < pl.deltaCteSqLowerTh_);
}

tuple<double, double> CrosstrackErrorEvaluator::Integrate(const double integralPeriod) {
  double integralTime = 0.0;
  double integralCteSq = 0.0;
  double integralDeltaCteSq = 0.0;
  for (auto histElem = history_.begin(); histElem != history_.end(); ++histElem) {
    double deltaTime;
    if ((integralTime + histElem->deltaTime) <= integralPeriod) {
      // Use full delta time for this element.
      deltaTime = histElem->deltaTime;
    } else {
      /* Integral time period exceeded, adjust delta time for this last element. */
      deltaTime = integralPeriod - integralTime;
      histElem = history_.end() - 1; // Skip to end.
    }
    integralTime += deltaTime;
    integralCteSq += histElem->cteSq * deltaTime;
    integralDeltaCteSq += histElem->deltaCteSq * deltaTime;
  }
  const double integralCteSqPerSecond = integralCteSq / integralTime;
  const double integralDeltaCteSqPerSecond = integralDeltaCteSq / integralTime;
  return make_tuple(integralCteSqPerSecond, integralDeltaCteSqPerSecond);
}

void CrosstrackErrorEvaluator::ReduceHistory() {
  double integralTime = 0.0;

  // Erase unnecessary history.
  for (auto histElem = history_.begin(); histElem != history_.end(); ++histElem) {
    integralTime += histElem->deltaTime;
    if (integralTime > longestIntegralPeriod_) {
      history_.erase(histElem + 1, history_.end());
      break;
    }
  }
}
