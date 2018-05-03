#ifndef HELPERS_H
#define HELPERS_H

#include <vector>

namespace Helpers {
  /** Convert degrees to radians. */
  double deg2rad(double x);

  /** Convert radians to degrees. */
  double rad2deg(double x);

  /** Calculates the euclidean distance. */
  double distance(double x1, double y1, double x2, double y2);

  /** Get closest waypoint. */
  int ClosestWaypoint(double x, double y, const std::vector<double> &maps_x, const std::vector<double> &maps_y);

  /** Get closes waypoint ahead of ego vehicle. */
  int NextWaypoint(double x, double y, double theta, const std::vector<double> &maps_x, const std::vector<double> &maps_y);

  /** Transform from Cartesian x,y coordinates to Frenet s,d coordinates */
  std::vector<double> getFrenet(double x, double y, double theta, const std::vector<double> &maps_x, const std::vector<double> &maps_y);

  /** Transform from Frenet s,d coordinates to Cartesian x,y */
  std::vector<double> getXY(double s, double d, const std::vector<double> &maps_s, const std::vector<double> &maps_x, const std::vector<double> &maps_y);
}

#endif /* HELPERS_H */
