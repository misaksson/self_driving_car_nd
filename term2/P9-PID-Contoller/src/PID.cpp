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
  p_error_ = 0.0;
  i_error_ = 0.0;
  d_error_ = 0.0;
  previous_cte_ = NAN;
  integral_cte_ = 0.0;
}

void PID::UpdateError(double cte) {
  // Proportional error
  p_error_ = Kp_ * cte;

  // Integral error
  integral_cte_ += cte;
  i_error_ = Ki_ * integral_cte_;

  // Derivative error
  if (!isnan(previous_cte_)) {
    const double diff_cte = cte - previous_cte_;
    d_error_ = Kd_ * diff_cte;
  }
  previous_cte_ = cte;
}

double PID::TotalError() {
  return p_error_ + i_error_ + d_error_;
}

