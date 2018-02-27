#include <iostream>
#include <cmath>
#include <algorithm>
#include <deque>

#include "vehicle_controller.h"

using namespace std;

VehicleController::VehicleController() {
  /* Setup RECOVERY control mode. */
  controllers_[RECOVERY].targetSpeed = 11.1760; // Meter per second (25 MPH)
  controllers_[RECOVERY].steering.Init(4.91194, 0.110417, 47.7133,
                                   false, "RecoverySteering", true);
  controllers_[RECOVERY].throttle.Init(1.01684, 0.00078523, 0.868024);

  /* Setup SAFE control mode. */
  controllers_[SAFE].targetSpeed = 13.4112; // Meter per second (30 MPH)
  controllers_[SAFE].steering.Init(4.91194, 0.110417, 47.7133,
                                   false, "SafeSteering", true);
  controllers_[SAFE].throttle.Init(1.01684, 0.00078523, 0.868024);

  /* Setup CAREFUL control mode */
  controllers_[CAREFUL].targetSpeed = 20.1168; // Meter per second (45 MPH)
  controllers_[CAREFUL].steering.Init(4.86809, 0.109132, 47.6244,
                                      false, "CarefulSteering", true);
  controllers_[CAREFUL].throttle.Init(0.707856, 0.000713845, 0.99455);

  /* Setup MODERATE control mode */
  controllers_[MODERATE].targetSpeed = 26.8224; // Meter per second (60 MPH)
  controllers_[MODERATE].steering.Init(4.86809, 0.109132, 47.6244,
                                       false, "ModerateSteering", true);
  controllers_[MODERATE].throttle.Init(0.707856, 0.000713845, 0.99455);

  /* Setup CHALLENGING control mode */
  controllers_[CHALLENGING].targetSpeed = 35.7632; // Meter per second (80 MPH)
  controllers_[CHALLENGING].steering.Init(4.86809, 0.109132, 47.6244,
                                          false, "ChallengingSteering", true);
  controllers_[CHALLENGING].throttle.Init(0.707856, 0.000713845, 0.99455);

  /* Setup BOLD control mode */
  controllers_[BOLD].targetSpeed = 44.7040; // Meter per second (100 MPH)
  controllers_[BOLD].steering.Init(4.86809, 0.109132, 47.6244,
                                   false, "BoldSteering", true);
  controllers_[BOLD].throttle.Init(0.707856, 0.000713845, 0.99455);


  /* Start in safe mode. */
  currentMode_ = SAFE;
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
  // Accumulate total error from all controllers.
  double totalSteeringError = 0.0;
  for (auto controller = controllers_.begin(); controller != controllers_.end(); ++controller) {
    totalSteeringError += controller->steering.GetAccumulatedError();
  }

  // Evaluate and update parameters for next tuning iteration.
  for (auto controller = controllers_.begin(); controller != controllers_.end(); ++controller) {
    controller->steering.SetNextParams(totalSteeringError);
    controller->throttle.SetNextParams();
  }
}
