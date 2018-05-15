#include "constants.h"
#include "helpers.h"
#include "path_planner.h"
#include "spline.h"
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

/** Helper function to print a path to terminal. */
static void printVector(string name, vector<double> xs, vector<double> ys);
/** Helper function to round values in the same way as the simulator interface. */
static double roundToSevenSignificantDigits(double value);

PathPlanner::PathPlanner(string waypointsMapFile, int pathLength) : numFinePathCoords(pathLength) {
  ifstream in_map_(waypointsMapFile.c_str(), ifstream::in);
  string line;
  while (getline(in_map_, line)) {
    istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }
}

PathPlanner::~PathPlanner() {
}

PathPlanner::Path PathPlanner::CalcNext(const VehicleData &vehicleData, const PathPlanner::Path &previousPath,
                                        double previousEnd_s, double previousEnd_d) {

  double targetSpeed = Logic(vehicleData);

  int numPreviousCoords = previousPath.x.size();

  /* Calculate a course vector of coordinates that the vehicle shall follow. */
  Path globalCoursePath;

  double localOffset_x, localOffset_y, localOffset_yaw, localOffset_s, localOffset_d;
  if (numPreviousCoords < 2) {
    /* Calculate next path from current vehicle state.
     * The local coordinate system is set at the vehicles current position.
     */
    localOffset_x = vehicleData.ego.x;
    localOffset_y = vehicleData.ego.y;
    localOffset_yaw = vehicleData.ego.yaw;
    localOffset_s = vehicleData.ego.s;
    localOffset_d = vehicleData.ego.d;
    speed = vehicleData.ego.speed;
    acceleration = 0.0;
    numPreviousCoords = 0;

    // Initialize the course path vector using current position, and also a extrapolated coordinate behind the vehicle.
    globalCoursePath.x.push_back(vehicleData.ego.x - cos(localOffset_yaw));
    globalCoursePath.y.push_back(vehicleData.ego.y - sin(localOffset_yaw));
    globalCoursePath.x.push_back(vehicleData.ego.x);
    globalCoursePath.y.push_back(vehicleData.ego.y);
  } else {
    /* Calculate next path by extending previous path.
     * The local coordinate system is set at end of previous path.
     */
    localOffset_x = previousPath.x[numPreviousCoords - 1];
    localOffset_y = previousPath.y[numPreviousCoords - 1];
    localOffset_yaw = atan2(previousPath.y[numPreviousCoords - 1] - previousPath.y[numPreviousCoords - 2],
                            previousPath.x[numPreviousCoords - 1] - previousPath.x[numPreviousCoords - 2]);
    localOffset_s = previousEnd_s;
    localOffset_d = previousEnd_d;

    // Initialize the course path vector using the two last positions in previous vector.
    globalCoursePath.x.push_back(previousPath.x[numPreviousCoords - 2]);
    globalCoursePath.y.push_back(previousPath.y[numPreviousCoords - 2]);
    globalCoursePath.x.push_back(previousPath.x[numPreviousCoords - 1]);
    globalCoursePath.y.push_back(previousPath.y[numPreviousCoords - 1]);
  }

  // The course path vector is further extended using Frenet coordinates.
  const double global_d = 1.5 * constants.laneWidth;
  vector<double> global_ss = {localOffset_s + 30.0,
                              localOffset_s + 60.0,
                              localOffset_s + 90.0};

  for (auto global_s = global_ss.begin(); global_s != global_ss.end(); ++global_s) {
    double global_x, global_y;
    tie(global_x, global_y) = Helpers::getXY(*global_s, global_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
    globalCoursePath.x.push_back(global_x);
    globalCoursePath.y.push_back(global_y);
  }

  // Transform the course vector of coordinates to a local coordinate system located at end of previous path.
  const int numCoursePathCoords = globalCoursePath.x.size();
  Path localCoursePath(numCoursePathCoords);
  for (int i = 0; i < numCoursePathCoords; ++i) {
    tie(localCoursePath.x[i], localCoursePath.y[i]) = Helpers::global2LocalTransform(globalCoursePath.x[i], globalCoursePath.y[i],
                                                                                     localOffset_x, localOffset_y, localOffset_yaw);
  }

  // Fit a spline to the local course path.
  tk::spline localPathSpline;
  localPathSpline.set_points(localCoursePath.x, localCoursePath.y);

  /* The spline is used to calculate a fine vector along the path of local coordinates. A linear approximation for the
   * spline is used to calculate the distance between each local x-coordinate, such that each step approximately will
   * equal the expected step length at target speed.
   */
  Path globalFinePath = previousPath;
  const int numExtendedCoords = (numFinePathCoords - numPreviousCoords);
  vector<double> deltaDistances;
  double extendedDistance;
  tie(deltaDistances, extendedDistance) = CalcDeltaDistances(numExtendedCoords, targetSpeed);

  if (numExtendedCoords > 0) {
    const double distanceFactor_x = extendedDistance / sqrt(pow(localPathSpline(extendedDistance), 2.0) + pow(extendedDistance, 2.0));
    double local_x = 0.0;
    for(int i = 0; i < numExtendedCoords; ++i) {
      local_x += deltaDistances[i] * distanceFactor_x;
      const double local_y = localPathSpline(local_x);
      double global_x, global_y;
      tie(global_x, global_y) = Helpers::local2GlobalTransform(local_x, local_y,
                                                               localOffset_x, localOffset_y, localOffset_yaw);
      globalFinePath.x.push_back(global_x);
      globalFinePath.y.push_back(global_y);
    }
  }
//  printSpeedAccJerk(globalFinePath, numExtendedCoords);
  return globalFinePath;
}

double PathPlanner::Logic(const VehicleData &vehicleData) {
  double targetSpeed = constants.speedLimit;
  double minLongitudinalDiff = HUGE_VAL;
  for (auto otherVehicle = vehicleData.others.begin(); otherVehicle != vehicleData.others.end(); ++otherVehicle) {
    cout << "id=" << otherVehicle->id <<
            ", x=" << otherVehicle->x <<
            ", y=" << otherVehicle->y <<
            ", vx=" << otherVehicle->vx <<
            ", vy=" << otherVehicle->vy <<
            ", s=" << otherVehicle->s <<
            ", d=" << otherVehicle->d;

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

tuple<vector<double>, double> PathPlanner::CalcDeltaDistances(int numDistances, const double targetSpeed) {
  vector<double> deltaDistances;
  double totalDistance = 0.0;

  for (int i = 0; i < numDistances; ++i) {
    /* Calculate the largest acceleration value (a0) from which it will be possible to settle the speed without
     * overshooting the targetSpeed. This is derived from integrating the acceleration decrease function that gives
     * the minimum delta speed change needed to reach acceleration = 0.
     * accelerationMin(t) = a0 - jerkLimit * t
     * deltaSpeedMin = integrate a0 - jerkLimit * t dt from 0 to T1 =
     *               = a0 * T1 - jerkLimit * T1^2 / 2
     * accelerationMin(T1) = a0 - jerkLimit * T1 = 0, gives
     * T1 = a0 / jerkLimit
     * Substituting T1 then results in
     * deltaSpeedMin = a0^2 / jerkLimit - jerkLimit * (a0 / jerkLimit)^2 / 2 =
     *               = a0^2 / (2 * jerkLimit)
     * The deltaSpeedMin is given, e.g. it's the remainingSpeedChange, so the settlingAcceleration (a0) is calculated as
     * a0 = sqrt(deltaSpeedMin * 2 * jerkLimit)
     *
     * To compensate for non-continuous function the settlingAccleration is further reduced by (jerkLimit * deltaTime) / 2.0.
     */
    double remainingSpeedChange = targetSpeed - speed;
    double settlingAcceleration = remainingSpeedChange > 0.0 ? sqrt(fabs(remainingSpeedChange) * 2.0 * constants.jerkLimit) - (constants.jerkLimit * constants.deltaTime) / 2.0 :
                                                              -sqrt(fabs(remainingSpeedChange) * 2.0 * constants.jerkLimit) + (constants.jerkLimit * constants.deltaTime) / 2.0;
    // Max possible value for next acceleration.
    double maxAcceleration = min(constants.accelerationLimit, acceleration + constants.jerkLimit * constants.deltaTime);
    // Min possible value for next acceleration.
    double minAcceleration = max(-constants.accelerationLimit, acceleration - constants.jerkLimit * constants.deltaTime);
    // Limit the wanted settling acceleration by what's actually possible.
    acceleration = max(minAcceleration, min(maxAcceleration, settlingAcceleration));

    // Update speed and calculate the delta distance for this delta time.
    speed += acceleration * constants.deltaTime;
    deltaDistances.push_back(speed * constants.deltaTime);
    totalDistance += deltaDistances.back();
  }
  return make_tuple(deltaDistances, totalDistance);
}

void PathPlanner::printSpeedAccJerk(PathPlanner::Path path, int num) {
  vector<double> speeds, accelerations, jerks;
  for (int i = 1; i < path.x.size(); ++i) {
    speeds.push_back(Helpers::distance(path.x[i - 1], path.y[i - 1], path.x[i], path.y[i]) / constants.deltaTime);
  }
  for (int i = 1; i < speeds.size(); ++i) {
    accelerations.push_back((speeds[i] - speeds[i - 1]) / constants.deltaTime);
  }
  for (int i = 1; i < accelerations.size(); ++i) {
    jerks.push_back((accelerations[i] - accelerations[i - 1]) / constants.deltaTime);
  }
  num = min(num, static_cast<int>(jerks.size()));
  auto speed = speeds.end() - num;
  auto acc = accelerations.end() - num;
  auto jerk = jerks.end() - num;
  for (int i = 0; i < num; ++i) {
    cout << "speed = " << *speed <<
            ", acc = " << *acc <<
            ", jerk = " << *jerk <<
            endl;
    speed++;
    acc++;
    jerk++;
  }
}

static void printVector(string name, vector<double> xs, vector<double> ys) {
  cout.precision(10);
  cout << name << "=" << fixed;
  for (int i = 0; i < xs.size(); ++i) {
    cout << "(" << xs[i] << ", " << ys[i] << ") ";
  }
  cout << endl;
}

static double roundToSevenSignificantDigits(double value) {
  std::stringstream lStream;
  lStream << setprecision(7) << value;
  return stod(lStream.str());
}

