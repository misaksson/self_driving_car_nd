#ifndef PATH_PLANNER_H
#define PATH_PLANNER_H

#include <string>
#include <vector>

class PathPlanner {
public:
  /** Constructor
   * @param waypointsMapFile A CSV file path containing track waypoints.
   * @param trackLength The length of the track.
   */
  PathPlanner(std::string waypointsMapFile, double trackLength);
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

private:
  /* Map values for waypoint's x,y,s and d normalized normal vectors that are
   * extracted from file during construction. */
  std::vector<double> map_waypoints_x;
  std::vector<double> map_waypoints_y;
  std::vector<double> map_waypoints_s;
  std::vector<double> map_waypoints_dx;
  std::vector<double> map_waypoints_dy;

  const double laneWidth = 4.0;
};

#endif /* PATH_PLANNER_H */
