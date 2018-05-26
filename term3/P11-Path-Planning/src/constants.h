#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "helpers.h"

class Constants {
private:
  /** Extra margin to compensate for numerical and approximation errors. */
  const double EXTRA_MARGIN = 0.2;

public:
  Constants() : speedLimit(Helpers::milesPerHour2MetersPerSecond(50.0) - EXTRA_MARGIN),
                accelerationLimit(10.0 - EXTRA_MARGIN),
                jerkLimit(10.0 - EXTRA_MARGIN),
                deltaTime(0.02),
                trackLength(6945.554),
                laneWidth(4.0),
                numLanes(3) {};
  virtual ~Constants() {};

  /** Speed limit in the simulator given in meter per second. */
  const double speedLimit;
  /** Acceleration limit in the simulator given in meter per second squared. */
  const double accelerationLimit;
  /** Jerk limit in the simulator given in meter per second cubed. */
  const double jerkLimit;
  /** Time in seconds between updates in the simulator. */
  const double deltaTime;
  /** Distance around the track. */
  const double trackLength;
  /** Lane width in meters. */
  const double laneWidth;
  /** Number of lanes available. */
  const int numLanes;

};

extern const Constants constants;

#endif /* CONSTANTS_H */
