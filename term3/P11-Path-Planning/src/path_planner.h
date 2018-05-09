#ifndef PATH_PLANNER_H
#define PATH_PLANNER_H

#include "helpers.h"
#include <string>
#include <tuple>
#include <vector>

class PathPlanner {
public:
  /** Constructor
   * @param waypointsMapFile A CSV file path containing track waypoints.
   * @param trackLength The length of the track.
   */
  PathPlanner(std::string waypointsMapFile, double trackLength, int pathLength);
  virtual ~PathPlanner();

  /** Localization data of ego vehicle. */
  struct EgoVehicleData {
    EgoVehicleData() {};
    EgoVehicleData(double x, double y, double s, double d, double yaw, double speed) :
                   x(x), y(y), s(s), d(d), yaw(yaw), speed(speed) {};
    double x;
    double y;
    double s;
    double d;
    double yaw;
    double speed;
  };

  /** Localization data of other vehicles. */
  struct OtherVehicleData {
    OtherVehicleData() {};
    OtherVehicleData(std::vector<double> data) : id(static_cast<uint64_t>(data[0])), x(data[1]), y(data[2]),
                                                 vx(data[3]), vy(data[4]), s(data[5]), d(data[6]) {};
    uint64_t id;
    double x;
    double y;
    double vx;
    double vy;
    double s;
    double d;
  };

  struct Path {
    Path() {};
    Path(int nCoords) {
      x.resize(nCoords);
      y.resize(nCoords);
    };
    Path(std::vector<double> path_x, std::vector<double> path_y) : x(path_x), y(path_y) {};

    std::vector<double> x;
    std::vector<double> y;
  };

  /** Calculate the path to follow.
   * The simulator will update the position of ego vehicle to the next in the list every 0.02 second.
   * @param input Input data, see PP_Input.
   * @param egoVehicle Ego vehicle localization data.
   * @param otherVehicles Other vehicles localization data.
   * @param previousPath Previously calculated path coordinates not yet visited by the simulator.
   * @return Next path coordinates for the simulator.
   */
   Path CalcNext(const EgoVehicleData &egoVehicle, const std::vector<OtherVehicleData> &otherVehicles, const Path &previousPath,
                 double previousEnd_s, double previousEnd_d);

  /* Map values for waypoint's x,y,s and d normalized normal vectors that are
   * extracted from file during construction.
   * Belongs to the public interface only for convenience. */
  std::vector<double> map_waypoints_x;
  std::vector<double> map_waypoints_y;
  std::vector<double> map_waypoints_s;
  std::vector<double> map_waypoints_dx;
  std::vector<double> map_waypoints_dy;

  /** Extra margin to compensate for numerical and approximation errors. */
  const double EXTRA_MARGIN = 0.2;
  /** Speed limit in the simulator given in meter per second. */
  const double speedLimit = Helpers::milesPerHour2MetersPerSecond(50.0) - EXTRA_MARGIN;
  /** Acceleration limit in the simulator given in meter per second squared. */
  const double accelerationLimit = 10.0 - EXTRA_MARGIN;
  /** Jerk limit in the simulator given in meter per second cubed. */
  const double jerkLimit = 10.0 - EXTRA_MARGIN;
  /** Time in seconds between updates in the simulator. */
  const double deltaTime = 0.02;

private:
  const int numFinePathCoords;
  const double laneWidth = 4.0; /**< Lane width in meters. */

  double speed; /**< Vehicle speed at end of calculated path. */
  double acceleration; /**< Vehicle acceleration at end of calculated path. */

  std::tuple<std::vector<double>, double> CalcDeltaDistances(int numDistances, const double targetSpeed);
  void printSpeedAccJerk(Path path, int num);
};

#endif /* PATH_PLANNER_H */
