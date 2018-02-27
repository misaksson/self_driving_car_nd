#ifndef VEHICLE_CONTROLLER_H
#define VEHICLE_CONTROLLER_H

#include <array>
#include <vector>
#include "PID.h"
#include "twiddle.h"

class VehicleController {
public:
  VehicleController();
  ~VehicleController();

  /** Available vehicle control modes. */
  enum ControlMode {
    RECOVERY = 0,
    SAFE,
    CAREFUL,
    MODERATE,
    CHALLENGING,
    BOLD,

    NUM_CONTROL_MODES,
  };
  /** Set the active vehicle control mode. */
  void SetMode(ControlMode mode);

  /** Calculate next steering value. */
  double CalcSteeringValue(double deltaTime, double speed, double cte);

  /** Calculate next throttle value. */
  double CalcThrottleValue(double deltaTime, double speed);

  /** Update the controller with next parameters to try out. */
  void SetNextParams();

  /** Gather the total error from all controllers.
   * Call this method, e.g. for each lap on the track. The errors are stored
   * internally and evaluated when preparing next parameter tuning. */
  void GatherError();

private:
  struct Controller {
    double targetSpeed;
    Twiddle steering;
    Twiddle throttle;
  };

  /** Vehicle controllers for each control mode. */
  std::array<Controller, NUM_CONTROL_MODES> controllers_;

  /** The active vehicle control mode. */
  ControlMode currentMode_;

  /** Number of times all controllers have been tuned. */
  int tuningCount_;

  /** The controller currently being tuned. */
  ControlMode currentlyTuning_;

  /** Gathered errors that are analyzed in preparation of next parameter tuning. */
  std::vector<double> gatheredErrors_;
};

#endif /* VEHICLE_CONTROLLER_H */
