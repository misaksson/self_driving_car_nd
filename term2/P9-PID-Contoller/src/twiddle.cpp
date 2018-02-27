#include "ansi_color_codes.h"
#include "twiddle.h"
#include <assert.h>
#include <cmath>
#include <iostream>

using namespace std;

Twiddle::Twiddle() {}
Twiddle::~Twiddle() {}

void Twiddle::Init(double Kp, double Ki, double Kd) {
  Init(Kp, Ki, Kd, false, "", false);
}

void Twiddle::Init(double Kp, double Ki, double Kd,
                   bool active, string name, bool consoleOutput) {
  double dKp = fabs(Kp * 0.1);
  double dKi = fabs(Ki * 0.1);
  double dKd = fabs(Kp * 0.1);
  Init(Kp, Ki, Kd, dKp, dKi, dKd, active, name, consoleOutput);
}

void Twiddle::Init(double Kp, double Ki, double Kd,
                   double dKp, double dKi, double dKd,
                   bool active, string name, bool consoleOutput) {
  PID::Init(Kp, Ki, Kd);
  dKp_ = dKp;
  dKi_ = dKi;
  dKd_ = dKp;
  active_ = active;
  name_ = name;
  consoleOutput_ = consoleOutput;
  lowestError_ = HUGE_VAL;
  nextTuning_ = INIT;
  currentCoefficient_ = 0;
  iteration_ = 0;
  Reset();
}


void Twiddle::Reset() {
  accumulatedError_ = 0.0;
  PID::Reset();
}

double Twiddle::CalcError(double cte) {
  accumulatedError_ += pow(cte, 2);
  return PID::CalcError(cte);
}

double Twiddle::GetAccumulatedError() {
  return accumulatedError_;
}

void Twiddle::SetNextParams() {
  // No additional external error
  SetNextParams(accumulatedError_);
}

void Twiddle::SetNextParams(double totalError) {
  if (active_) {
    double *p[] = {&Kp_, &Ki_, &Kd_};
    double *dp[] = {&dKp_, &dKi_, &dKd_};

    if (totalError < lowestError_) {
      // This tuning is the best so far.
      lowestError_ = totalError;

      if (consoleOutput_) {
        cout << ANSI::GREEN << name_ << " " << iteration_ << " " << totalError
          << " Kp=" << Kp_ << " Ki=" << Ki_ << " Kd=" << Kd_
          << " dKp=" << dKp_ << " dKi=" << dKi_ << " dKd=" << dKd_
          << " " << currentCoefficient_<< ANSI::RESET << endl;
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
        cout << ANSI::RED << name_ << " " << iteration_ << " " << totalError
          << " Kp=" << Kp_ << " Ki=" << Ki_ << " Kd=" << Kd_
          << " dKp=" << dKp_ << " dKi=" << dKi_ << " dKd=" << dKd_
          << " " << currentCoefficient_ << ANSI::RESET << endl;
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
    ++iteration_;
  }

  // Always reset, even if twiddle not is active for this controller.
  Reset();
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
