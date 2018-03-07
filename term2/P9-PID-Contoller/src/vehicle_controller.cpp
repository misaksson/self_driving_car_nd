#include <iostream>
#include <cmath>
#include <algorithm>
#include <deque>

#include "vehicle_controller.h"

using namespace std;

const VehicleController::ControlMode startMode = VehicleController::MODERATE;

VehicleController::VehicleController() {
  /* Start in safe mode. */
  currentMode_ = SAFE;

  /* Tuning initialization. */
  tuningCount_ = 0;
  currentlyTuning_ = startMode;

  /* Setup RECOVERY control mode. */
  controllers_[RECOVERY].targetSpeed = 11.1760; // Meter per second (25 MPH)
  controllers_[RECOVERY].steering.Init(3.47135, 0.116486, 44.8846, 0.526819, 0.00216197, 0.431034,
                                       currentlyTuning_ == RECOVERY, "RecoverySteering", true);
  controllers_[RECOVERY].throttle.Init(1.01684, 0.00078523, 0.868024);

  /* Setup SAFE control mode. */
  controllers_[SAFE].targetSpeed = 13.4112; // Meter per second (30 MPH)
  controllers_[SAFE].steering.Init(7.27161, 0.0983717, 48.6137, 0.288544, 0.00326224, 0.193157,
                                   currentlyTuning_ == SAFE, "SafeSteering", true);
  controllers_[SAFE].throttle.Init(1.01684, 0.00078523, 0.868024);

  /* Setup CAREFUL control mode */
  controllers_[CAREFUL].targetSpeed = 20.1168; // Meter per second (45 MPH)
  controllers_[CAREFUL].steering.Init(4.25341, 0.104686, 52.6885, 0.795786, 0.00442287, 0.651097,
                                      currentlyTuning_ == CAREFUL, "CarefulSteering", true);
  controllers_[CAREFUL].throttle.Init(0.707856, 0.000713845, 0.99455);

  /* Setup MODERATE control mode */
  controllers_[MODERATE].targetSpeed = 26.8224; // Meter per second (60 MPH)
  controllers_[MODERATE].steering.Init(2.5618, 0.10703, 48.1941, 1.18877, 0.0098697, 0.723442,
                                       currentlyTuning_== MODERATE, "ModerateSteering", true);
  controllers_[MODERATE].throttle.Init(0.707856, 0.000713845, 0.99455);

  /* Setup CHALLENGING control mode */
  controllers_[CHALLENGING].targetSpeed = 35.7632; // Meter per second (80 MPH)
  controllers_[CHALLENGING].steering.Init(3.17781, 0.161733, 47.9923, 0.811945, 0.00915465, 0.902161,
                                          currentlyTuning_ == CHALLENGING, "ChallengingSteering", true);
  controllers_[CHALLENGING].throttle.Init(0.707856, 0.000713845, 0.99455);

  /* Setup BOLD control mode */
  controllers_[BOLD].targetSpeed = 44.7040; // Meter per second (100 MPH)
  controllers_[BOLD].steering.Init(8.63782, 0.152766, 48.363, 0.893139, 0.00600636, 0.664318,
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

  // Calculate cost values mean and stdev
  const double sum = std::accumulate(costValues_.begin(), costValues_.end(), 0.0);
  const double mean = sum / costValues_.size();
  vector<double> diff(costValues_.size());
  transform(costValues_.begin(), costValues_.end(), diff.begin(), [mean](double x) { return x - mean; });
  const double sq_sum = inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  const double stdev = sqrt(sq_sum / costValues_.size());

  // Calculated cost values median
  nth_element(costValues_.begin(), costValues_.begin() + costValues_.size() / 2, costValues_.end());
  const double median = costValues_[costValues_.size() / 2];
  costValues_.clear();

  cout << "Mean: " << mean << endl;
  cout << "Stdev: " << stdev << endl;
  cout << "Median: " << median << endl;

  // Evaluate and update parameters for next tuning iteration.
  for (auto controller = controllers_.begin(); controller != controllers_.end(); ++controller) {
    controller->steering.SetNextParams(mean + stdev);
    controller->throttle.SetNextParams();
  }

  if (controllers_[currentlyTuning_].steering.GetTuningCount() > tuningCount_) {
    /* Tune next controller. */
    controllers_[currentlyTuning_].steering.Abort();
    currentlyTuning_ = static_cast<ControlMode>((currentlyTuning_ + 1) % NUM_CONTROL_MODES);
    controllers_[currentlyTuning_].steering.Continue();
    if (currentlyTuning_ == startMode) {
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
  costValues_.push_back(lapTime);
}
