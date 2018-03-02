#include <iostream>
#include <cmath>
#include <algorithm>
#include <deque>

#include "vehicle_controller.h"

using namespace std;

VehicleController::VehicleController() {
  /* Start in safe mode. */
  currentMode_ = SAFE;

  /* Tuning initialization. */
  tuningCount_ = 0;
  currentlyTuning_ = RECOVERY;

  /* Setup RECOVERY control mode. */
  controllers_[RECOVERY].targetSpeed = 11.1760; // Meter per second (25 MPH)
  controllers_[RECOVERY].steering.Init(2.99833, 0.117943, 47.614,
                                       0.513833, 0.00575124, 0.697799,
                                       currentlyTuning_ == RECOVERY, "RecoverySteering", true);
  controllers_[RECOVERY].throttle.Init(1.01684, 0.00078523, 0.868024);

  /* Setup SAFE control mode. */
  controllers_[SAFE].targetSpeed = 13.4112; // Meter per second (30 MPH)
  controllers_[SAFE].steering.Init(6.07426, 0.107155, 48.0287,
                                   0.378368, 0.00569372, 0.28143,
                                   currentlyTuning_ == SAFE, "SafeSteering", true);
  controllers_[SAFE].throttle.Init(1.01684, 0.00078523, 0.868024);

  /* Setup CAREFUL control mode */
  controllers_[CAREFUL].targetSpeed = 20.1168; // Meter per second (45 MPH)
  controllers_[CAREFUL].steering.Init(5.40832, 0.10714, 48.8081,
                                      0.691568, 0.00516756, 0.51439,
                                      currentlyTuning_ == CAREFUL, "CarefulSteering", true);
  controllers_[CAREFUL].throttle.Init(0.707856, 0.000713845, 0.99455);

  /* Setup MODERATE control mode */
  controllers_[MODERATE].targetSpeed = 26.8224; // Meter per second (60 MPH)
  controllers_[MODERATE].steering.Init(2.74351, 0.117893, 46.5252,
                                       0.853788, 0.00857715, 0.776171,
                                       currentlyTuning_ == MODERATE, "ModerateSteering", true);
  controllers_[MODERATE].throttle.Init(0.707856, 0.000713845, 0.99455);

  /* Setup CHALLENGING control mode */
  controllers_[CHALLENGING].targetSpeed = 35.7632; // Meter per second (80 MPH)
  controllers_[CHALLENGING].steering.Init(5.40352, 0.135581, 47.9793,
                                          0.519586, 0.00866379, 0.425116,
                                          currentlyTuning_ == CHALLENGING, "ChallengingSteering", true);
  controllers_[CHALLENGING].throttle.Init(0.707856, 0.000713845, 0.99455);

  /* Setup BOLD control mode */
  controllers_[BOLD].targetSpeed = 44.7040; // Meter per second (100 MPH)
  controllers_[BOLD].steering.Init(6.75553, 0.13746, 46.0841,
                                   0.776171, 0.00953017, 0.577318,
                                   currentlyTuning_ == BOLD, "BoldSteering", true);
  controllers_[BOLD].throttle.Init(0.707856, 0.000713845, 0.99455);

}

VehicleController::~VehicleController() {}

void VehicleController::SetMode(VehicleController::ControlMode mode) {
  // Propagate internal PID controller state.
  controllers_[mode].steering.SetState(controllers_[currentMode_].steering);
  currentMode_ = mode;
}

double VehicleController::CalcSteeringValue(double deltaTime, double speed, double cte) {
  /* Calculate steering_value. This should depend on the speed, for now
   * it's just set inversely proportional. */
  double steerValue = -controllers_[currentMode_].steering.CalcError(cte) / speed;

  // Saturate steerValue to valid range [-1, 1]
  steerValue = max(-1.0, min(1.0, steerValue));

  return steerValue;
}

double VehicleController::CalcThrottleValue(double deltaTime, double speed) {
  const double targetSpeed = controllers_[currentMode_].targetSpeed;
  const double speedError = speed - targetSpeed;
  const double throttleValue = -controllers_[currentMode_].throttle.CalcError(speedError);
  return throttleValue;
}

void VehicleController::SetNextParams() {
  if (costValues_.size() == 0) {
    SetCost();
  }

  // Calculated the median error
  nth_element(costValues_.begin(), costValues_.begin() + costValues_.size() / 2, costValues_.end());
  const double medianError = costValues_[costValues_.size() / 2];
  costValues_.clear();
  cout << "Median error: " << medianError << endl;

  // Evaluate and update parameters for next tuning iteration.
  for (auto controller = controllers_.begin(); controller != controllers_.end(); ++controller) {
    controller->steering.SetNextParams(medianError);
    controller->throttle.SetNextParams();
  }

  if (controllers_[currentlyTuning_].steering.GetTuningCount() > tuningCount_) {
    /* Tune next controller. */
    controllers_[currentlyTuning_].steering.Abort();
    currentlyTuning_ = static_cast<ControlMode>((currentlyTuning_ + 1) % NUM_CONTROL_MODES);
    controllers_[currentlyTuning_].steering.Continue();
    if (currentlyTuning_ == 0) {
      // All controllers tuned an iteration.
      ++tuningCount_;
    }
  }
}

void VehicleController::SetCost() {
  // Accumulate errors from all controllers and reset.
  double totalSteeringError = 0.0;
  for (auto controller = controllers_.begin(); controller != controllers_.end(); ++controller) {
    totalSteeringError += controller->steering.GetAccumulatedError();
    controller->steering.Reset();
  }
  cout << "Gathered error: " << totalSteeringError << endl;
  costValues_.push_back(totalSteeringError);
}

void VehicleController::SetCost(double lapTime) {
  cout << "Lap time: " << lapTime << endl;
  costValues_.push_back(lapTime);
}
