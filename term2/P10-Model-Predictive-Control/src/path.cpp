#include <assert.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>

#include "Eigen-3.3/Eigen/Dense"
#include "path.h"
#include "polynomial.h"

using namespace std;

Path::Path(string filename) {
  string line;

  // Open CSV file
  ifstream waypointsFile(filename);
  assert(waypointsFile.is_open());

  // Read CSV header
  getline(waypointsFile, line);
  assert(line == "x,y");

  // Read CSV data
  while (getline(waypointsFile, line)) {
    size_t idx;
    waypointsX_.push_back(stod(line, &idx));
    waypointsY_.push_back(stod(line.substr(idx + 1)));
  }
  waypointsFile.close();
  nWaypoints_ = waypointsX_.size();
}

Path::~Path() {}

/** Transform a global coordinate to the local coordinate system of the vehicle
 * at current location.
 * @param globalX Global coordinate to transform.
 * @param globalY Global coordinate to transform.
 * @param vehicleX Global coordinate of the vehicle.
 * @param vehicleY Global coordinate of the vehicle.
 * @param vehiclePhi Global orientation of the vehicle.
 * @output The transformed x and y coordinates.
 * */
inline tuple<double, double> global2LocalTransform(const double globalX, const double globalY,
                                                   const double vehicleX, const double vehicleY,
                                                   const double vehiclePhi) {

  Eigen::Matrix3d local2Global;
  local2Global << cos(vehiclePhi), -sin(vehiclePhi), vehicleX,
                  sin(vehiclePhi), cos(vehiclePhi), vehicleY,
                  0.0, 0.0, 1.0;
  Eigen::Matrix3d global2Local = local2Global.inverse();
  Eigen::Vector3d global(globalX, globalY, 1.0);
  Eigen::Vector3d local = global2Local * global;
  return make_tuple(local[0], local[1]);
}

Eigen::VectorXd Path::GetPoly(const double vehicleX, const double vehicleY, const double vehiclePhi) {
  // Find waypoint at shortest distance from vehicle
  double shortestDistance = HUGE_VAL;
  size_t shortestDistanceIdx = 0;
  for (size_t i = 0; i < nWaypoints_; ++i) {
    const double distance = sqrt(pow(waypointsX_[i] - vehicleX, 2.0) +
                                 pow(waypointsY_[i] - vehicleY, 2.0));
    if (distance < shortestDistance) {
      shortestDistance = distance;
      shortestDistanceIdx = i;
    }
  }

  /* Extract a few waypoints neighboring the one at shortest distance, and
   * transform them to a local coordinate system for the vehicle.
   */
  const size_t nWaypointsToFitBefore = 4;
  const size_t nWaypointsToFitAfter = 4;
  const size_t nWaypointsToFit = nWaypointsToFitBefore + 1 + nWaypointsToFitAfter;

  Eigen::VectorXd localX(nWaypointsToFit), localY(nWaypointsToFit);
  for (size_t localIdx = 0; localIdx < nWaypointsToFit; ++localIdx) {
    size_t waypointsIdx = (shortestDistanceIdx - nWaypointsToFitBefore + localIdx + nWaypoints_) % nWaypoints_;
    tie(localX[localIdx], localY[localIdx]) = global2LocalTransform(waypointsX_[waypointsIdx], waypointsY_[waypointsIdx],
                                                                    vehicleX, vehicleY, vehiclePhi);
  }

  // Fit a polynomial to the waypoints given in vehicle local coordinates
  Eigen::VectorXd coeffs = Polynomial::Fit(localX, localY, 3);
  return coeffs;
}

Eigen::VectorXd Path::GetPoly(const vector<double> waypointsX, const vector<double> waypointsY,
                              const double vehicleX, const double vehicleY, const double vehiclePhi) {
  const size_t nWaypointsToFit = waypointsX.size();
  Eigen::VectorXd localX(nWaypointsToFit), localY(nWaypointsToFit);
  for (size_t i = 0; i < nWaypointsToFit; ++i) {
    tie(localX[i], localY[i]) = global2LocalTransform(waypointsX[i], waypointsY[i], vehicleX, vehicleY, vehiclePhi);
  }

  // Fit a polynomial to the waypoints given in vehicle local coordinates
  Eigen::VectorXd coeffs = Polynomial::Fit(localX, localY, 3);
  return coeffs;
}
