#include "../constants.h"
#include "trajectory.h"
#include "../helpers.h"
#include "../spline.h"
#include "../vehicle_data.h"
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace std;

/** Calculates delta distances when accelerating to target speed.
 * The acceleration is done as effectively as possible within the allowed acceleration and jerk limits.
 */
static tuple<vector<double>, double> CalcDeltaDistances(double speed, double acceleration, const double targetSpeed);

Path::Trajectory::Trajectory() {};
Path::Trajectory::Trajectory(int nCoords) {
  x.resize(nCoords);
  y.resize(nCoords);
};
Path::Trajectory::Trajectory(std::vector<double> x, std::vector<double> y) : x(x), y(y) {};

int Path::Trajectory::size() const {
  return x.size();
}

VehicleData::EgoVehicleData Path::Trajectory::getEndState(const Helpers &helpers) const {
  VehicleData::EgoVehicleData result;
  if (size() > 1) {
    result.x = x[size() - 1];
    result.y = y[size() - 1];
    result.yaw = atan2(y[size() - 1] - y[size() - 2],
                       x[size() - 1] - x[size() - 2]);
    tie(result.s, result.d) = helpers.getFrenet(result.x, result.y, result.yaw);
    result.speed = sqrt(pow(y[size() - 1] - y[size() - 2], 2.0) +
                        pow(x[size() - 1] - x[size() - 2], 2.0)) / constants.deltaTime;
  }
  return result;
}

Path::Trajectory::Kinematics Path::Trajectory::getKinematics() const {
  Kinematics kinematics;
  for (int i = 1; i < size(); ++i) {
    kinematics.speeds.push_back(Helpers::distance(x[i - 1], y[i - 1], x[i], y[i]) / constants.deltaTime);
    kinematics.yaws.push_back(atan2(y[i] - y[i - 1], x[i] - x[i - 1]));
  }
  for (int i = 1; i < size() - 1; ++i) {
    kinematics.accelerations.push_back((kinematics.speeds[i] - kinematics.speeds[i - 1]) / constants.deltaTime);
    kinematics.yawRates.push_back((kinematics.yaws[i] - kinematics.yaws[i - 1]) / constants.deltaTime);
  }
  for (int i = 1; i < size() - 2; ++i) {
    kinematics.jerks.push_back((kinematics.accelerations[i] - kinematics.accelerations[i - 1]) / constants.deltaTime);
  }
  return kinematics;
}

Path::TrajectoryCalculator::TrajectoryCalculator(const Helpers &helpers) : helpers(helpers) {}

Path::TrajectoryCalculator::~TrajectoryCalculator() {};

Path::Trajectory Path::TrajectoryCalculator::Accelerate(const VehicleData::EgoVehicleData &start, double delta_speed) const {
  Trajectory globalCourse;

  globalCourse.x.push_back(start.x - cos(start.yaw));
  globalCourse.y.push_back(start.y - sin(start.yaw));
  globalCourse.x.push_back(start.x);
  globalCourse.y.push_back(start.y);

  // The course path vector is further extended using Frenet coordinates.
  const double global_d = start.d;
  vector<double> global_ss = {start.s + 30.0,
                              start.s + 60.0,
                              start.s + 90.0};

  for (auto global_s = global_ss.begin(); global_s != global_ss.end(); ++global_s) {
    double global_x, global_y;
    tie(global_x, global_y) = helpers.getXY(*global_s, global_d);
    globalCourse.x.push_back(global_x);
    globalCourse.y.push_back(global_y);
  }
  // Transform the course vector of coordinates to a local coordinate system located at the start position.
  Trajectory localCourse(globalCourse.x.size());
  for (int i = 0; i < globalCourse.x.size(); ++i) {
    tie(localCourse.x[i], localCourse.y[i]) = Helpers::global2LocalTransform(globalCourse.x[i], globalCourse.y[i],
                                                                             start.x, start.y, start.yaw);
  }

  // Fit a spline to the local course coordinates.
  tk::spline localSpline;
  localSpline.set_points(localCourse.x, localCourse.y);

  /* The spline is used to calculate a fine vector along the trajectory of local coordinates. A linear approximation
   * for the spline is used to calculate the distance between each local x-coordinate, such that each step
   * approximately will equal the expected step length at current speed.
   */
  Trajectory globalFine;
  vector<double> deltaDistances;;
  double distance;
  tie(deltaDistances, distance) = CalcDeltaDistances(start.speed, 0.0, start.speed + delta_speed);
  if (deltaDistances.size() > 0) {
    const double distanceFactor_x = distance / sqrt(pow(localSpline(distance), 2.0) + pow(distance, 2.0));
    double local_x = 0.0;
    for(int i = 0; i < deltaDistances.size(); ++i) {
      local_x += deltaDistances[i] * distanceFactor_x;
      const double local_y = localSpline(local_x);
      double global_x, global_y;
      tie(global_x, global_y) = Helpers::local2GlobalTransform(local_x, local_y,
                                                               start.x, start.y, start.yaw);
      globalFine.x.push_back(global_x);
      globalFine.y.push_back(global_y);
    }
  }
  return globalFine;
}

static tuple<vector<double>, double> CalcDeltaDistances(double speed, double acceleration, const double targetSpeed) {
  vector<double> deltaDistances;
  double totalDistance = 0.0;
  const bool increaseSpeed = speed < targetSpeed;
  const double tolerance = 0.1;

  while ((increaseSpeed && (speed < (targetSpeed - tolerance))) ||
         (!increaseSpeed && (speed > (targetSpeed + tolerance)))) {
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
    double settlingAcceleration = remainingSpeedChange > 0.0 ?
       sqrt(fabs(remainingSpeedChange) * 2.0 * constants.jerkLimit) - (constants.jerkLimit * constants.deltaTime) / 2.0 :
      -sqrt(fabs(remainingSpeedChange) * 2.0 * constants.jerkLimit) + (constants.jerkLimit * constants.deltaTime) / 2.0;

    // Max possible value for next acceleration.
    double maxAcceleration = min(constants.accelerationLimit,
                                 acceleration + constants.jerkLimit * constants.deltaTime);
    // Min possible value for next acceleration.
    double minAcceleration = max(-constants.accelerationLimit,
                                 acceleration - constants.jerkLimit * constants.deltaTime);
    // Limit the wanted settling acceleration by what's actually possible.
    acceleration = max(minAcceleration, min(maxAcceleration, settlingAcceleration));

    // Update speed and calculate the delta distance for this delta time.
    speed += acceleration * constants.deltaTime;
    deltaDistances.push_back(speed * constants.deltaTime);
    totalDistance += deltaDistances.back();
  }
  return make_tuple(deltaDistances, totalDistance);
}

Path::Trajectory Path::TrajectoryCalculator::ConstantSpeed(const VehicleData::EgoVehicleData &start, int numCoords) const {
  Trajectory globalCourse;

  globalCourse.x.push_back(start.x - cos(start.yaw));
  globalCourse.y.push_back(start.y - sin(start.yaw));
  globalCourse.x.push_back(start.x);
  globalCourse.y.push_back(start.y);

  // The course path vector is further extended using Frenet coordinates.
  const double global_d = start.d;
  vector<double> global_ss = {start.s + 30.0,
                              start.s + 60.0,
                              start.s + 90.0};

  for (auto global_s = global_ss.begin(); global_s != global_ss.end(); ++global_s) {
    double global_x, global_y;
    tie(global_x, global_y) = helpers.getXY(*global_s, global_d);
    globalCourse.x.push_back(global_x);
    globalCourse.y.push_back(global_y);
  }

  // Transform the course vector of coordinates to a local coordinate system located at the start position.
  Trajectory localCourse(globalCourse.x.size());
  for (int i = 0; i < globalCourse.x.size(); ++i) {
    tie(localCourse.x[i], localCourse.y[i]) = Helpers::global2LocalTransform(globalCourse.x[i], globalCourse.y[i],
                                                                             start.x, start.y, start.yaw);
  }

  // Fit a spline to the local course coordinates.
  tk::spline localSpline;
  localSpline.set_points(localCourse.x, localCourse.y);

  /* The spline is used to calculate a fine vector along the trajectory of local coordinates. A linear approximation
   * for the spline is used to calculate the distance between each local x-coordinate, such that each step
   * approximately will equal the expected step length at current speed.
   */
  Trajectory globalFine;
  const double distance = sqrt(pow(globalCourse.x.back() - start.x, 2.0) + pow(globalCourse.y.back() - start.y, 2.0));
  const double distanceFactor_x = distance / sqrt(pow(localSpline(distance), 2.0) + pow(distance, 2.0));
  const double deltaDistance = start.speed * constants.deltaTime;

  for(int i = 0; i < numCoords; ++i) {
    const double local_x = (i + 1) * deltaDistance * distanceFactor_x;
    const double local_y = localSpline(local_x);
    double global_x, global_y;
    tie(global_x, global_y) = Helpers::local2GlobalTransform(local_x, local_y,
                                                             start.x, start.y, start.yaw);
    globalFine.x.push_back(global_x);
    globalFine.y.push_back(global_y);
  }
  return globalFine;
}

Path::Trajectory Path::TrajectoryCalculator::AdjustSpeed(const VehicleData::EgoVehicleData &start, double delta_s, double delta_d, double delta_speed) const {
  Trajectory globalCourse;

  globalCourse.x.push_back(start.x - cos(start.yaw));
  globalCourse.y.push_back(start.y - sin(start.yaw));
  globalCourse.x.push_back(start.x);
  globalCourse.y.push_back(start.y);

  double middle_x, middle_y;
  tie(middle_x, middle_y) = helpers.getXY(start.s + (delta_s / 2.0), start.d + (delta_d / 2.0));
  globalCourse.x.push_back(middle_x);
  globalCourse.y.push_back(middle_y);

  double end_x, end_y;
  tie(end_x, end_y) = helpers.getXY(start.s + delta_s, start.d + delta_d);
  globalCourse.x.push_back(end_x);
  globalCourse.y.push_back(end_y);

  double extrapolated_x, extrapolated_y;
  tie(extrapolated_x, extrapolated_y) = helpers.getXY(start.s + delta_s + 10.0, start.d + delta_d);
  globalCourse.x.push_back(extrapolated_x);
  globalCourse.y.push_back(extrapolated_y);

  // Transform the course vector of coordinates to a local coordinate system located at the start position.
  Trajectory localCourse(globalCourse.x.size());
  for (int i = 0; i < globalCourse.x.size(); ++i) {
    tie(localCourse.x[i], localCourse.y[i]) = Helpers::global2LocalTransform(globalCourse.x[i], globalCourse.y[i],
                                                                             start.x, start.y, start.yaw);
  }

  // Fit a spline to the local course coordinates.
  tk::spline localSpline;
  localSpline.set_points(localCourse.x, localCourse.y);

  /* The spline is used to calculate a fine vector along the trajectory of local coordinates. A linear approximation
   * for the spline is used to calculate the distance between each local x-coordinate, such that each step
   * approximately will equal the expected step length at current speed.
   */
  Trajectory globalFine;
  const double distance = sqrt(pow(end_x - start.x, 2.0) + pow(end_y - start.y, 2.0));
  const double distanceFactor_x = distance / sqrt(pow(localSpline(distance), 2.0) + pow(distance, 2.0));
  const double distance_x = distance * distanceFactor_x;
  double local_x = 0.0;
  do {

      const double speed = start.speed + (delta_speed * local_x / distance_x);
      const double deltaDistance = speed * constants.deltaTime;
      const double deltaDistance_x = deltaDistance * distanceFactor_x;
      local_x += deltaDistance_x;
      const double local_y = localSpline(local_x);
      double global_x, global_y;
      tie(global_x, global_y) = Helpers::local2GlobalTransform(local_x, local_y,
                                                               start.x, start.y, start.yaw);
      globalFine.x.push_back(global_x);
      globalFine.y.push_back(global_y);
  } while (local_x < distance_x);

  return globalFine;
}
