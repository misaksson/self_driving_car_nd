#include <iostream>
#include <cmath>
#include <algorithm>
#include <deque>

#include "vehicle_controller.h"

using namespace std;

VehicleController::VehicleController() {
  /* Setup safe mode. */
  tune_[SAFE] = false;
  controllers_[SAFE].targetSpeed = 15.0; // Meter per second
  controllers_[SAFE].steering = Twiddle();
  controllers_[SAFE].steering.Init(4.91194, 0.110417, 47.7133,
                                   0.00936508, 9.45968e-05, 0.00851371,
                                   tune_[SAFE], "SafeSteering", true);
  controllers_[SAFE].throttle = Twiddle();
  controllers_[SAFE].throttle.Init(1.01684, 0.00078523, 0.868024,
                                   0.0490737, 2.21772e-05, 0.0401512,
                                   false, "SafeThrottle", true);

  /* Setup normal mode */
  tune_[NORMAL] = false;
  controllers_[NORMAL].targetSpeed = 28.0; // Meter per second
  controllers_[NORMAL].steering = Twiddle();
  controllers_[NORMAL].steering.Init(4.86809, 0.109132, 47.6244,
                                     1.29692e-05, 1.07632e-06, 1.44102e-05,
                                     tune_[NORMAL], "NormalSteering", true);
  controllers_[NORMAL].throttle = Twiddle();
  controllers_[NORMAL].throttle.Init(0.707856, 0.000713845, 0.99455,
                                     false, "NormalThrottle", true);

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
    controllers_[controlMode].throttle.SetNextParams();
    controllers_[controlMode].steering.Reset();
  }
}
