#include <iostream>
#include <cmath>
#include <algorithm>
#include <deque>
#include <sstream>

#include "cte_eval.h"

using namespace std;

const string ANSI_RED = "\033[;31m";
const string ANSI_GREEN = "\033[;32m";
const string ANSI_YELLOW = "\033[;33m";
const string ANSI_RESET = "\033[0m";

CrosstrackErrorEvaluator::CrosstrackErrorEvaluator() : isOk_(true), previousCte_(0.0) {};
CrosstrackErrorEvaluator::~CrosstrackErrorEvaluator() {};

bool CrosstrackErrorEvaluator::IsOk(double deltaTime, double cte) {
  stringstream debugstream;
  history_.push_front({.deltaTime = deltaTime,
                       .cteSq = pow(cte, 2.0),
                       .deltaCteSq = pow(cte - previousCte_, 2.0)});
  previousCte_ = cte;

  double integralTime = 0.0;
  double integralCteSq = 0.0;
  double integralDeltaCteSq = 0.0;
  const double integralPeriod = isOk_ ? triggerPeriod_ : releasePeriod_;

  auto histElem = history_.begin();
  for (; histElem != history_.end(); ++histElem) {
    integralTime += histElem->deltaTime;
    double deltaTime;
    if (integralTime <= integralPeriod) {
      // Use full delta time for this element.
      deltaTime = histElem->deltaTime;
    } else {
      /* Integral time period exceeded, adjust delta time for this last
       * element. */
      deltaTime = integralPeriod - (integralTime - histElem->deltaTime);
      break;
    }
    integralCteSq += histElem->cteSq * deltaTime;
    integralDeltaCteSq += histElem->deltaCteSq * deltaTime;
  }

  // Erase unnecessary history.
  for (; histElem != history_.end(); ++histElem) {
    if (integralTime > max(triggerPeriod_, releasePeriod_)) {
    //if (integralTime > triggerPeriod_) {
      history_.erase(histElem + 1, history_.end());
      break;
    }
  }

  const double integralCteSqPerSecond = integralCteSq / integralPeriod;
  const double integralDeltaCteSqPerSecond = integralDeltaCteSq / integralPeriod;


  debugstream.setf(ios::fixed, ios::floatfield);
  debugstream.precision(8);
  if (isOk_) {
    // Only one threshold must be passed to set state not OK.
    if ((integralCteSqPerSecond > cteSqUpperTh_) ||
        (integralDeltaCteSqPerSecond > deltaCteSqUpperTh_)) {
      isOk_ = false;
      debugstream << ANSI_RED << "NOK: ";
    } else {
      debugstream << ANSI_GREEN << "OK: ";
    }
    debugstream << (integralCteSqPerSecond > cteSqUpperTh_ ? ANSI_RED : ANSI_YELLOW) << integralCteSqPerSecond << " ";
    debugstream << (integralDeltaCteSqPerSecond > deltaCteSqUpperTh_ ? ANSI_RED : ANSI_YELLOW) << integralDeltaCteSqPerSecond << " ";
  } else {
    // Both thresholds must be passed to set state OK.
    if ((integralCteSqPerSecond < cteSqLowerTh_) &&
        (integralDeltaCteSqPerSecond < deltaCteSqLowerTh_)) {
      isOk_ = true;
      debugstream << ANSI_GREEN << "OK: ";
    } else {
      debugstream << ANSI_RED << "NOK: ";
    }
    debugstream << (integralCteSqPerSecond < cteSqLowerTh_ ? ANSI_GREEN : ANSI_YELLOW) << integralCteSqPerSecond << " ";
    debugstream << (integralDeltaCteSqPerSecond < deltaCteSqLowerTh_ ? ANSI_GREEN : ANSI_YELLOW) << integralDeltaCteSqPerSecond << " ";
  }
  debugstream << ANSI_RESET << endl;

#ifdef DEBUG_PRINT
  cout << debugstream.str();
#endif

  return isOk_;
}

