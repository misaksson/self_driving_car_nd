#include "twiddle.h"
#include <cmath>
#include <iostream>

using namespace std;

const string ANSI_RED = "\033[;31m";
const string ANSI_GREEN = "\033[;32m";
const string ANSI_RESET = "\033[0m";


Twiddle::Twiddle() : consoleOutput_(true) {}
Twiddle::Twiddle(bool consoleOutput) : consoleOutput_(consoleOutput) {}

Twiddle::~Twiddle() {}

void Twiddle::Init(double Kp, double Ki, double Kd) {
  PID::Init(Kp, Ki, Kd);
}

void Twiddle::Init(double Kp, double Ki, double Kd,
                   double dKp, double dKi, double dKd, bool active) {
  Init(Kp, Ki, Kd);
  active_ = active;
  if (active_) {
    active_ = active;
    dKp_ = dKp;
    dKi_ = dKi;
    dKd_ = dKp;
    lowestError_ = HUGE_VAL;
    accumulatedError_ = 0.0;
    nextTuning_ = INIT;
    currentCoefficient_ = 0;
    iteration_ = 0;
  }
}

void Twiddle::Init(double Kp, double Ki, double Kd, bool active) {
  double dKp = fabs(Kp * 0.1);
  double dKi = fabs(Ki * 0.1);
  double dKd = fabs(Kp * 0.1);
  Init(Kp, Ki, Kd, dKp, dKi, dKd, active);
}

double Twiddle::CalcError(double cte) {
  accumulatedError_ += fabs(cte);
  return PID::CalcError(cte);
}

void Twiddle::SetNextParams() {
  if (active_) {
    double *p[] = {&Kp_, &Ki_, &Kd_};
    double *dp[] = {&dKp_, &dKi_, &dKd_};

    if (accumulatedError_ < lowestError_) {
      // This tuning is the best so far.
      lowestError_ = accumulatedError_;

      if (consoleOutput_) {
        cout << ANSI_GREEN << iteration_ << " " << accumulatedError_ << " Kp=" << Kp_ << " Ki=" << Ki_ << " Kd=" << Kd_
          << " dKp=" << dKp_ << " dKi=" << dKi_ << " dKd=" << dKd_ << " " << currentCoefficient_<< ANSI_RESET << endl;
      }

      if (nextTuning_ != INIT) {
        // Try a more aggressive tuning next time.
        *dp[currentCoefficient_] *= 1.1;

        // Continue by increasing next coefficient.
        nextTuning_ = INCREASE;
      }
    } else {
      // No improvement
      if (consoleOutput_) {
        cout << ANSI_RED << iteration_ << " " << accumulatedError_ << " Kp=" << Kp_ << " Ki=" << Ki_ << " Kd=" << Kd_
          << " dKp=" << dKp_ << " dKi=" << dKi_ << " dKd=" << dKd_ << " " << currentCoefficient_ << ANSI_RESET << endl;
      }
    }


    // Adjust parameters for next run.
    switch (nextTuning_) {
      case REVERT:
        // Revert current coefficient tuning attempt.
        *p[currentCoefficient_] += (*dp[currentCoefficient_]);

        // Try a less aggressive tuning next time.
        *dp[currentCoefficient_] *= 0.9;

        // Note: Intentional fall-through (no break).
      case INCREASE:
        currentCoefficient_ = (currentCoefficient_ + 1) % 3;

        // Note: Intentional fall-through (no break).
      case INIT:
        *p[currentCoefficient_] += (*dp[currentCoefficient_]);
        nextTuning_ = DECREASE; // If the increase fails, then decrease next iteration.
        break;

      case DECREASE:
        *p[currentCoefficient_] -= 2.0 * (*dp[currentCoefficient_]);
        nextTuning_ = REVERT; // If the decrease also fails, then revert and continue with next coefficient.
        break;
    };

    // Prepare internal state for next run.
    accumulatedError_ = 0.0;
    Reset();
    ++iteration_;
  }
}

void Twiddle::Abort() {
  active_ = false;

  // Revert ongoing tuning attempt.
  double *p[] = {&Kp_, &Ki_, &Kd_};
  double *dp[] = {&dKp_, &dKi_, &dKd_};
  switch (nextTuning_) {
    case REVERT:
      // Ongoing decrease.
      *p[currentCoefficient_] += (*dp[currentCoefficient_]);
      break;
    case DECREASE:
      // Ongoing increase.
      *p[currentCoefficient_] -= (*dp[currentCoefficient_]);
      break;
    default:
      /* Otherwise do nothing. */
      break;
  };
  nextTuning_ = INIT;
  accumulatedError_ = 0.0;
}

void Twiddle::Continue() {
  active_ = true;
}
