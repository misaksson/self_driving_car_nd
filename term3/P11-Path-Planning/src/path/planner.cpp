#include "../constants.h"
#include "../helpers.h"
#include "../spline.h"
#include "planner.h"
#include "trajectory.h"
#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace std;

/** Helper function to round values in the same way as the simulator interface. */
static double roundToSevenSignificantDigits(double value);

Path::Planner::Planner(const Helpers &helpers, const Path::TrajectoryCalculator &trajectoryCalculator, int pathLength) :
    helpers(helpers), trajectoryCalculator(trajectoryCalculator), pathLength(pathLength) {}

Path::Planner::~Planner() {
}

Path::Trajectory Path::Planner::CalcNext(const VehicleData &vehicleData, const Path::Trajectory &previousTrajectory,
                                       double previousEnd_s, double previousEnd_d) {
  double targetSpeed = Logic(vehicleData);
  int numPreviousCoords = previousTrajectory.x.size();
  VehicleData::EgoVehicleData localRef;
  if (previousTrajectory.size() < 2) {
    localRef = vehicleData.ego;
  } else {
    localRef = previousTrajectory.getEndState(helpers);
  }
  Path::Trajectory output = previousTrajectory + trajectoryCalculator.Accelerate(localRef, targetSpeed - localRef.speed);

  localRef = output.getEndState(helpers);
  if (output.size() < pathLength) {
    output += trajectoryCalculator.ConstantSpeed(localRef, pathLength - output.size());
  }
  return output;
}

double Path::Planner::Logic(const VehicleData &vehicleData) {
  double targetSpeed = constants.speedLimit;
  double minLongitudinalDiff = HUGE_VAL;
  for (auto otherVehicle = vehicleData.others.begin(); otherVehicle != vehicleData.others.end(); ++otherVehicle) {
    cout << *otherVehicle;

    bool isSameLane = vehicleData.ego.d < otherVehicle->d + constants.laneWidth / 2.0 &&
                      vehicleData.ego.d > otherVehicle->d - constants.laneWidth / 2.0;
    cout << (isSameLane ? " Same lane ": " Another lane ");
    double longitudinalDiff;
    if ((otherVehicle->s < 1000.0) && (vehicleData.ego.s > (constants.trackLength - 1000.0))) {
      // Other vehicle has wrapped around the track.
      longitudinalDiff = otherVehicle->s + (constants.trackLength - vehicleData.ego.s);
    } else if ((vehicleData.ego.s < 1000.0) && (otherVehicle->s > (constants.trackLength - 1000.0))) {
      // Ego vehicle has wrapped around the track.
      longitudinalDiff = vehicleData.ego.s + (constants.trackLength - otherVehicle->s);
    } else {
      // No wrap around to consider.
      longitudinalDiff = otherVehicle->s - vehicleData.ego.s;
    }

    bool isAhead = longitudinalDiff > 0.0;
    cout << fabs(longitudinalDiff) << " meters " << (isAhead ? " ahead" : " behind") << " of egoVehicle";
    double otherVehicleSpeed = sqrt(pow(otherVehicle->vx, 2.0) + pow(otherVehicle->vy, 2.0));
    double speedDiff = otherVehicleSpeed - vehicleData.ego.speed;
    bool isSlower = speedDiff < 0.0;
    cout << " at a speed that is " << fabs(speedDiff) << " m/s " << (isSlower ? "slower " : "faster");

    if (isSameLane && isAhead) {
      /** In Sweden this is the recommended distance to a vehicle ahead, measured in seconds. */
      const double recommendedLongitudinalTimeDiff = 3.0;
      const double criticalLongitudinalTimeDiff = 2.0;
      double recommendedLongitudinalDiff = vehicleData.ego.speed * recommendedLongitudinalTimeDiff;
      double criticalLongitudinalDiff = vehicleData.ego.speed * criticalLongitudinalTimeDiff;
      if (longitudinalDiff < recommendedLongitudinalDiff &&
          longitudinalDiff < minLongitudinalDiff) {
        if (longitudinalDiff < criticalLongitudinalDiff) {
          targetSpeed = min(targetSpeed, otherVehicleSpeed * 0.9);
          cout << " -> speed set to critical " << targetSpeed;
        } else {
          targetSpeed = min(targetSpeed, otherVehicleSpeed);
          cout << " -> speed set to " << targetSpeed;
        }
        minLongitudinalDiff = longitudinalDiff;
      }
    }
    cout << endl;
  }
  return targetSpeed;
}

static double roundToSevenSignificantDigits(double value) {
  std::stringstream lStream;
  lStream << setprecision(7) << value;
  return stod(lStream.str());
}

