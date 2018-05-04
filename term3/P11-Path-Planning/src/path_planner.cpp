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
  /* Calculate a course vector of coordinates ahead of the vehicle in the middle lane. This is implemented using
   * Frenet coordinates which then are transformed to Cartesian coordinates. */
  const double global_d = 1.5 * laneWidth;
  const int numCoursePathCoords = 4;
  array<double, numCoursePathCoords> global_ss = {egoVehicle.s + 0.0,
                                                  egoVehicle.s + 30.0,
                                                  egoVehicle.s + 60.0,
                                                  egoVehicle.s + 90.0};
  Path globalCoursePath(numCoursePathCoords);
  for (int i = 0; i < numCoursePathCoords; ++i) {
    tie(globalCoursePath.x[i], globalCoursePath.y[i]) = Helpers::getXY(global_ss[i], global_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
  }

  // Transform the course vector of coordinates to a vehicle local coordinate system.
  Path localCoursePath(numCoursePathCoords);
  for (int i = 0; i < numCoursePathCoords; ++i) {
    tie(localCoursePath.x[i], localCoursePath.y[i]) = Helpers::global2LocalTransform(globalCoursePath.x[i], globalCoursePath.y[i],
                                                                                     egoVehicle.x, egoVehicle.y, egoVehicle.yaw);
  }

  // Fit a spline to the local course path.
  tk::spline localPathSpline;
  localPathSpline.set_points(localCoursePath.x, localCoursePath.y);

  /* Calculate a fine vector of local coordinates using the spline. For now, the distance between each point is
   * constant in vehicle local x-direction, which actually makes the vehicle speed up in curves. */
  const int numFinePathCoords = 50;
  Path globalFinePath(numFinePathCoords);
  for(int i = 0; i < numFinePathCoords; ++i) {
    double local_x = 0.25 * (double)(i + 1);
    double local_y = localPathSpline(local_x);
    tie(globalFinePath.x[i], globalFinePath.y[i]) = Helpers::local2GlobalTransform(local_x, local_y,
                                                                                   egoVehicle.x, egoVehicle.y, egoVehicle.yaw);
  }

  // Smooth out the transition from previous to next path.
  const int numPrevious = previousPath.x.size();
  for (int i = 0; i < numPrevious; ++i) {
    double smoothFactor = (double)i / (double)(numPrevious);
    globalFinePath.x[i] = globalFinePath.x[i] * smoothFactor + previousPath.x[i] * (1.0 - smoothFactor);
    globalFinePath.y[i] = globalFinePath.y[i] * smoothFactor + previousPath.y[i] * (1.0 - smoothFactor);
  }
  return globalFinePath;
}
