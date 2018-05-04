#ifndef HELPERS_H
#define HELPERS_H

#include <vector>
#include <tuple>

namespace Helpers {
  /** Convert degrees to radians. */
  double deg2rad(double x);

  /** Convert radians to degrees. */
  double rad2deg(double x);

  double milesPerHour2MetersPerSecond(double x);

  /** Calculates the euclidean distance. */
  double distance(double x1, double y1, double x2, double y2);

  /** Get closest waypoint. */
  int ClosestWaypoint(double x, double y, const std::vector<double> &maps_x, const std::vector<double> &maps_y);

  /** Get closes waypoint ahead of ego vehicle. */
  int NextWaypoint(double x, double y, double theta, const std::vector<double> &maps_x, const std::vector<double> &maps_y);

  /** Transform from Cartesian x,y coordinates to Frenet s,d coordinates */
  std::tuple<double, double>  getFrenet(double x, double y, double theta, const std::vector<double> &maps_x, const std::vector<double> &maps_y);

  /** Transform from Frenet s,d coordinates to Cartesian x,y */
  std::tuple<double, double>  getXY(double s, double d, const std::vector<double> &maps_s, const std::vector<double> &maps_x, const std::vector<double> &maps_y);

  std::tuple<double, double> global2LocalTransform(const double global_x, const double global_y,
                                                   const double localOffset_x, const double localOffset_y,
                                                   const double localOffset_psi);

  std::tuple<double, double> local2GlobalTransform(const double local_x, const double local_y,
                                                   const double localOffset_x, const double localOffset_y,
                                                   const double localOffset_psi);
}

#endif /* HELPERS_H */
