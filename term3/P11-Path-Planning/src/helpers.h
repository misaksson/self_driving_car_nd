#ifndef HELPERS_H
#define HELPERS_H

#include <string>
#include <vector>
#include <tuple>

class Helpers {
public:
  Helpers(std::string waypointsMapFile);
  virtual ~Helpers();

  /** Convert degrees to radians. */
  static double deg2rad(double x);

  /** Convert radians to degrees. */
  static double rad2deg(double x);

  static double milesPerHour2MetersPerSecond(double x);

  /** Calculates the euclidean distance. */
  static double distance(double x1, double y1, double x2, double y2);

  /** Calculates the yaw angle in radians */
  static double CalcYaw(double x1, double y1, double x2, double y2);

  /** Calculate lane number from Frenet d coordinate. */
  static int GetLane(double d);
  /** Calculate lane number from Cartesian coordinates. */
  static int GetLane(double x, double y, double theta);

  /** Calculates the longitudinal difference s1 - s2 while considering track wrap-around. */
  static double calcLongitudinalDiff(double s1, double s2);

  /** Get closest waypoint. */
  int ClosestWaypoint(double x, double y) const;

  /** Get closes waypoint ahead of ego vehicle. */
  int NextWaypoint(double x, double y, double theta) const;

  /** Transform from Cartesian x,y coordinates to Frenet s,d coordinates */
  std::tuple<double, double>  getFrenet(double x, double y, double theta) const;

  /** Transform from Frenet s,d coordinates to Cartesian x,y */
  std::tuple<double, double>  getXY(double s, double d) const;

  static std::tuple<double, double> global2LocalTransform(const double global_x, const double global_y,
                                                          const double localOffset_x, const double localOffset_y,
                                                          const double localOffset_psi);

  static std::tuple<double, double> local2GlobalTransform(const double local_x, const double local_y,
                                                          const double localOffset_x, const double localOffset_y,
                                                          const double localOffset_psi);
  /* Map values for waypoint's x,y,s and d normalized normal vectors that are
   * extracted from file during construction.
   * Belongs to the public interface only for convenience. */
  std::vector<double> map_waypoints_x;
  std::vector<double> map_waypoints_y;
  std::vector<double> map_waypoints_s;
  std::vector<double> map_waypoints_dx;
  std::vector<double> map_waypoints_dy;
};

/* Must be defined e.g. in main.h */
extern const Helpers helpers;

#endif /* HELPERS_H */
