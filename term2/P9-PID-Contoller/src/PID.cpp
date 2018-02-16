#include "PID.h"
#include <cmath>
#include <iostream>

using namespace std;

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
  Kp_ = Kp;
  Ki_ = Ki;
  Kd_ = Kd;
  previous_cte_ = NAN;
  integral_cte_ = 0.0;
}

double PID::CalcError(double cte) {
  // Proportional error
  const double p_error = Kp_ * cte;

  // Integral error
  integral_cte_ += cte;
  const double i_error = Ki_ * integral_cte_;

  // Derivative error
  const double diff_cte = isnan(previous_cte_) ? 0.0 : cte - previous_cte_;
  const double d_error = Kd_ * diff_cte;

  previous_cte_ = cte;

  const double totalError = p_error + i_error + d_error;
  return totalError;
}
