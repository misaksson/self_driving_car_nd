#ifndef VEHICLE_CONTROLLER_H
#define VEHICLE_CONTROLLER_H

#include "twiddle.h"

class VehicleController {
private:
  struct Controller {
    double targetSpeed;
    Twiddle steering;
    Twiddle throttle;
  };

  enum ControlMode {
    SAFE = 0,
    NORMAL,

    NUM_CONTROL_MODES,
  };
  Controller controllers_[NUM_CONTROL_MODES];
  bool tune_[NUM_CONTROL_MODES];
  ControlMode currentMode_;

public:
  VehicleController();
  ~VehicleController();
  void SetSafeMode();
  void SetNormalMode();
  double CalcSteeringValue(double deltaTime, double speed, double cte);
  double CalcThrottleValue(double deltaTime, double speed);
  /** Update the controller with next parameters to try out.
   * This method have no effect when tuning not is active. */
  void SetNextParams();

};

#endif /* VEHICLE_CONTROLLER_H */
