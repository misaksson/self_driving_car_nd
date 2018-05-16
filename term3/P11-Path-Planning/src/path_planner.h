#ifndef PATH_PLANNER_H
#define PATH_PLANNER_H

#include "helpers.h"
#include "vehicle_data.h"
#include <string>
#include <tuple>
#include <vector>

class PathPlanner {
public:
  /** Constructor
   * @param waypointsMapFile A CSV file path containing track waypoints.
   * @param pathLength Number of coordinates to send to simulator.
   */
  PathPlanner(const Helpers &helpers, int pathLength);
  virtual ~PathPlanner();

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
   Path CalcNext(const VehicleData &vehicleData, const Path &previousPath,
                 double previousEnd_s, double previousEnd_d);


private:
  const Helpers &helpers;
  const int numFinePathCoords;

  double speed; /**< Vehicle speed at end of calculated path. */
  double acceleration; /**< Vehicle acceleration at end of calculated path. */
  double Logic(const VehicleData &vehicleData);
  std::tuple<std::vector<double>, double> CalcDeltaDistances(int numDistances, const double targetSpeed);
  void printSpeedAccJerk(Path path, int num);
};

#endif /* PATH_PLANNER_H */
