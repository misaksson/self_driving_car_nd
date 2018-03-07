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

  /** Fits a third order polynomial to waypoints representing the road nearby
   * the vehicle. The polynomial is calculated in the vehicles local coordinate
   * system.
   * @param vehicleX Vehicle x in global coordinates.
   * @param vehicleY Vehicle y in global coordinates.
   * @param vehiclePsi Vehicle yaw angle in global coordinates.
   * @output Third order polynomial describing the road in vehicle coordinates.
   */
  Eigen::VectorXd GetPoly(const double vehicleX, const double vehicleY, const double vehiclePsi);

  /** Fits a third order polynomial to provided waypoints.The polynomial is
   * calculated in the vehicles local coordinate system.
   * @param waypointsX Global x coordinates of the waypoints to be fitted.
   * @param waypointsX Global y coordinates of the waypoints to be fitted.
   * @param vehicleX Vehicle x in global coordinates.
   * @param vehicleY Vehicle y in global coordinates.
   * @param vehiclePsi Vehicle yaw angle in global coordinates.
   * @output Third order polynomial describing the road in vehicle coordinates.
   */
  Eigen::VectorXd GetPoly(const std::vector<double> waypointsX, const std::vector<double> waypointsY,
                          const double vehicleX, const double vehicleY, const double vehiclePhi);

private:
  /** Number of waypoints. */
  size_t nWaypoints_;
  /** Waypoints x coordinates. */
  std::vector<double> waypointsX_;
  /** Waypoints y coordinates. */
  std::vector<double> waypointsY_;
};

#endif /* PATH_H */
