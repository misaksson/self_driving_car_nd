#include "helpers.h"
#include "path_planner.h"
#include "spline.h"
#include <array>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <vector>

using namespace std;

PathPlanner::PathPlanner(string waypointsMapFile, double trackLength) {
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

static void printVector(string name, vector<double> xs, vector<double> ys) {
  cout << name << "=";
  for (int i = 0; i < xs.size(); ++i) {
    cout << "(" << xs[i] << ", " << ys[i] << ") ";
  }
  cout << endl;
}

PathPlanner::Path PathPlanner::CalcNext(const PathPlanner::EgoVehicleData &egoVehicle, const vector<PathPlanner::OtherVehicleData> &otherVehicles, const PathPlanner::Path &previousPath,
                                        double previousEnd_s, double previousEnd_d) {
  int numPreviousCoords = previousPath.x.size();

  /* Calculate a course vector of coordinates that the vehicle shall follow. */
  Path globalCoursePath;

  double localOffset_x, localOffset_y, localOffset_yaw, localOffset_s, localOffset_d;
  if (numPreviousCoords < 2) {
    /* Calculate next path from current vehicle state.
     * The local coordinate system is set at the vehicles current position.
     */
    localOffset_x = egoVehicle.x;
    localOffset_y = egoVehicle.y;
    localOffset_yaw = egoVehicle.yaw;
    localOffset_s = egoVehicle.s;
    localOffset_d = egoVehicle.d;
    numPreviousCoords = 0;

    // Initialize the course path vector using current position, and also a extrapolated coordinate behind the vehicle.
    globalCoursePath.x.push_back(egoVehicle.x - cos(localOffset_yaw));
    globalCoursePath.y.push_back(egoVehicle.y - sin(localOffset_yaw));
    globalCoursePath.x.push_back(egoVehicle.x);
    globalCoursePath.y.push_back(egoVehicle.y);
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
  const double global_d = 1.5 * laneWidth;
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
  const int numFinePathCoords = 50;
  Path globalFinePath = previousPath;
  const double deltaDistance = 0.25;
  const int numExtendedCoords = (numFinePathCoords - numPreviousCoords);
  if (numExtendedCoords > 0) {

    const double extendedDistance = numExtendedCoords * deltaDistance;
    const double extendedDistance_x = pow(extendedDistance, 2.0) / sqrt(pow(localPathSpline(extendedDistance), 2.0) + pow(extendedDistance, 2.0));
    const double deltaDistance_x = extendedDistance_x / numExtendedCoords;
    for(int i = numPreviousCoords; i < numFinePathCoords; ++i) {
      double local_x = deltaDistance_x * (double)(i - numPreviousCoords + 1);
      double local_y = localPathSpline(local_x);
      double global_x, global_y;
      tie(global_x, global_y) = Helpers::local2GlobalTransform(local_x, local_y,
                                                               localOffset_x, localOffset_y, localOffset_yaw);
      globalFinePath.x.push_back(global_x);
      globalFinePath.y.push_back(global_y);
    }
  }

  return globalFinePath;
}
