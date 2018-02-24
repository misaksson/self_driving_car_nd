#include <iostream>
#include <cmath>
#include <algorithm>
#include <deque>

#include "vehicle_controller.h"

using namespace std;

VehicleController::VehicleController(bool tuneSafeMode, bool tuneNormalMode) {
  /* Setup safe mode. */
  tune_[SAFE] = tuneSafeMode;
  controllers_[SAFE].targetSpeed = 15.0; // Meter per second
  controllers_[SAFE].steering = Twiddle();
  controllers_[SAFE].steering.Init(4.868533823999999, 0.10715951136, 47.62272416,
                                   0.1, 0.001, 0.1,
                                   tuneSafeMode, "SafeSteering", true);
  controllers_[SAFE].throttle = Twiddle();
  controllers_[SAFE].throttle.Init(0.707856, 0.000713845, 0.99455, false);


  /* Setup normal mode */
  tune_[NORMAL] = tuneNormalMode;
  controllers_[NORMAL].targetSpeed = 30.0; // Meter per second
  controllers_[NORMAL].steering = Twiddle();
  controllers_[NORMAL].steering.Init(4.868533823999999, 0.10715951136, 47.62272416,
                                   0.1, 0.001, 0.1,
                                   tuneNormalMode, "NormalSteering", true);
  controllers_[NORMAL].throttle = Twiddle();
  controllers_[NORMAL].throttle.Init(0.707856, 0.000713845, 0.99455, false);

  /* Start in safe mode. */
  currentMode_ = SAFE;
}

VehicleController::~VehicleController() {}

void VehicleController::SetSafeMode() {
  currentMode_ = SAFE;
}

void VehicleController::SetNormalMode() {
  currentMode_ = NORMAL;
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
  double throttleValue = -controllers_[currentMode_].throttle.CalcError(speed - controllers_[currentMode_].targetSpeed);
  return throttleValue;
}

void VehicleController::SetNextParams() {
  if (tune_[SAFE]) {
    double externalError = controllers_[NORMAL].steering.GetAccumulatedError();
    controllers_[SAFE].steering.SetNextParams(externalError);
  }
  if (tune_[NORMAL]) {
    double externalError = controllers_[SAFE].steering.GetAccumulatedError();
    controllers_[NORMAL].steering.SetNextParams(externalError);
  }
  for (int controlMode = SAFE; controlMode < NUM_CONTROL_MODES; ++controlMode) {
    controllers_[controlMode].steering.Reset();
  }
}
