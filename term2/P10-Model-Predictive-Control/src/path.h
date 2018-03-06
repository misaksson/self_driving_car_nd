#ifndef PATH_H
#define PATH_H

#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"

class Path {
public:
  /** Constructor
   * @param filename Path to CSV file containing waypoints.
   */
  Path(std::string filename);
  ~Path();

  /** Fits a third order polygon to waypoints representing the road nearby the
   * vehicle. The polygon is calculated in the vehicles local coordinate
   * system.
   * @param vehicleX Vehicle x in global coordinates.
   * @param vehicleY Vehicle y in global coordinates.
   * @param vehiclePsi Vehicle yaw angle in global coordinates.
   * @output Third order polygon describing the road in vehicle coordinates.
   */
  Eigen::VectorXd GetPoly(double vehicleX, double vehicleY, double vehiclePsi);

private:
  /** Number of waypoints. */
  size_t nWaypoints_;
  /** Waypoints x coordinates. */
  std::vector<double> waypointsX_;
  /** Waypoints y coordinates. */
  std::vector<double> waypointsY_;
};

#endif /* PATH_H */
