#include "catch.hpp"
#include "../src/path.h"
#include "../src/polynomial.h"
#include <iostream>
#include <cmath>
#include <random>
#include <vector>

using namespace std;

TEST_CASE("Path should go around the track", "[path]") {
  Path path("../lake_track_waypoints.csv");

  const double startX = 179.3083, startY = 98.67102;
  const double velocity = 1.0;
  double direction = M_PI / 2.0;
  double x = startX, y = startY;
  double distance = 0;

  // Travel around the track until reaching start position again.
  do {
    Eigen::VectorXd coeffs = path.GetPoly(x, y, direction);
    double dydx = Polynomial::Evaluate(Polynomial::Derivative(coeffs), 0.0);
    double steeringAngle = atan(dydx);
    direction += steeringAngle;
    x += velocity * cos(direction);
    y += velocity * sin(direction);
    distance += velocity;
  } while (fabs(x - startX) > 0.5 || fabs(y - startY) > 0.5);

  /** The lap distance was measured during the PID controller project. */
  const double distancePerLap = 8000./7.;
  REQUIRE(distance == Approx(distancePerLap).margin(11.0));
}
