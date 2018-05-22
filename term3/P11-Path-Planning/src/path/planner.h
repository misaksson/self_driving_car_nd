#ifndef PLANNER_H
#define PLANNER_H

#include "../helpers.h"
#include "../vehicle_data.h"
#include "trajectory.h"
#include <string>
#include <tuple>
#include <vector>

namespace Path {
  class Planner {
  public:
    Planner(int pathLength);
    virtual ~Planner();

    /** Calculate the path to follow.
     * The simulator will update the position of ego vehicle to the next in the list every 0.02 second.
     * @param input Input data, see PP_Input.
     * @param egoVehicle Ego vehicle localization data.
     * @param otherVehicles Other vehicles localization data.
     * @param previousTrajectory Previously calculated trajectory coordinates not yet visited by the simulator.
     * @return Next path coordinates for the simulator.
     */
     Path::Trajectory CalcNext(const VehicleData &vehicleData, const Path::Trajectory &previousTrajectory);

  private:
    const int pathLength; /**< Number of coordinates to send to simulator. */
    double Logic(const VehicleData &vehicleData);

  }; /* class Planner */
}; /* namespace Path */
#endif /* PLANNER_H */