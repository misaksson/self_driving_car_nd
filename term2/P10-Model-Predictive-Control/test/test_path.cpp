#include "catch.hpp"
#include "../src/MPC.h"
#include "../src/path.h"
#include "../src/polynomial.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

using namespace std;

class Waypoints {
public:
  Waypoints() {

    string line;

    // Open CSV file
    ifstream waypointsFile("../lake_track_waypoints.csv");
    assert(waypointsFile.is_open());

    // Read CSV header
    getline(waypointsFile, line);
    assert(line == "x,y");

    // Read CSV data
    while (getline(waypointsFile, line)) {
      size_t idx;
      allWaypointsX_.push_back(stod(line, &idx));
      allWaypointsY_.push_back(stod(line.substr(idx + 1)));
    }
    waypointsFile.close();
  }
  ~Waypoints() {}

  /** Extracts a few waypoints near the vehicle position.
   * This is similar to the functionality providing waypoints in the simulator.
   * @output Waypoints near the vehicle. */
  tuple<vector<double>, vector<double>> GetNear(double vehicleX, double vehicleY) {
    // Find waypoint at shortest distance from vehicle
    double shortestDistance = HUGE_VAL;
    size_t shortestDistanceIdx = 0;
    for (size_t i = 0; i < allWaypointsX_.size(); ++i) {
      const double distance = sqrt(pow(allWaypointsX_[i] - vehicleX, 2.0) +
                                   pow(allWaypointsY_[i] - vehicleY, 2.0));
      if (distance < shortestDistance) {
        shortestDistance = distance;
        shortestDistanceIdx = i;
      }
    }

    /* Extract a few waypoints neighboring the one at shortest distance. */
    const size_t nWaypointsToFitBefore = 4;
    const size_t nWaypointsToFitAfter = 4;
    const size_t nWaypointsToFit = nWaypointsToFitBefore + 1 + nWaypointsToFitAfter;

    vector<double> extractedX, extractedY;

    for (size_t extractedIdx = 0; extractedIdx < nWaypointsToFit; ++extractedIdx) {
      size_t waypointsIdx = (shortestDistanceIdx - nWaypointsToFitBefore + extractedIdx + allWaypointsX_.size()) % allWaypointsX_.size();
      extractedX.push_back(allWaypointsX_[waypointsIdx]);
      extractedY.push_back(allWaypointsY_[waypointsIdx]);
    }
    return make_tuple(extractedX, extractedY);
  }

  vector<double> allWaypointsX_;
  vector<double> allWaypointsY_;
};

TEST_CASE("Path should go around the track", "[path]") {
   Waypoints waypoints;

  const double startX = waypoints.allWaypointsX_[0], startY = waypoints.allWaypointsY_[0];
  MPC::State globalState = {.x = startX, .y = startY, .psi = M_PI / 2.0, .v = 1.0};
  double distance = 0;

  // Travel around the track until reaching start position again.
  do {
    vector<double> waypointsX, waypointsY;
    tie(waypointsX, waypointsY) = waypoints.GetNear(globalState.x, globalState.y);
    Path path(waypointsX, waypointsY, globalState);
    MPC::State localState = path.GetLocalState();
    globalState.psi += localState.epsi;
    globalState.x += globalState.v * cos(globalState.psi);
    globalState.y += globalState.v * sin(globalState.psi);
    distance += globalState.v;
  } while (fabs(globalState.x - startX) > 0.5 || fabs(globalState.y - startY) > 0.5);

  /** The lap distance was measured during the PID controller project. */
  const double distancePerLap = 8000./7.;
  REQUIRE(distance == Approx(distancePerLap).margin(11.0));
}
