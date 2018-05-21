
#ifndef PATH_PLANNER_H
#define PATH_PLANNER_H

#include "helpers.h"
#include "vehicle_data.h"
#include "path/trajectory.h"
#include <string>
#include <tuple>
#include <vector>

class PathPlanner {
public:
  /** Constructor
   * @param waypointsMapFile A CSV file path containing track waypoints.
   * @param pathLength Number of coordinates to send to simulator.
   */
  PathPlanner(const Helpers &helpers, const Path::TrajectoryCalculator &trajectoryCalculator, int pathLength);
  virtual ~PathPlanner();

  /** Calculate the path to follow.
   * The simulator will update the position of ego vehicle to the next in the list every 0.02 second.
   * @param input Input data, see PP_Input.
   * @param egoVehicle Ego vehicle localization data.
   * @param otherVehicles Other vehicles localization data.
   * @param previousTrajectory Previously calculated trajectory coordinates not yet visited by the simulator.
   * @return Next path coordinates for the simulator.
   */
   Path::Trajectory CalcNext(const VehicleData &vehicleData, const Path::Trajectory &previousTrajectory,
                             double previousEnd_s, double previousEnd_d);

private:
  const Helpers &helpers;
  const Path::TrajectoryCalculator &trajectoryCalculator;
  const int pathLength; /**< Number of coordinates to send to simulator. */
  double Logic(const VehicleData &vehicleData);
};

#endif /* PATH_PLANNER_H */
