#ifndef PLANNER_H
#define PLANNER_H

#include "../helpers.h"
#include "../vehicle_data.h"
#include "logic.h"
#include "predict.h"
#include "trajectory.h"
#include <string>
#include <tuple>
#include <vector>

namespace Path {
  class Planner {
  public:
    Planner(int minTrajectoryLength, int maxTrajectoryLength);
    virtual ~Planner();

    /** Calculate the path to follow.
     * The simulator will update the position of ego vehicle to the next in the list every 0.02 second.
     * @param input Input data, see PP_Input.
     * @param egoVehicle Ego vehicle localization data.
     * @param otherVehicles Other vehicles localization data.
     * @param simulatorTrajectory Previously calculated trajectory coordinates not yet visited by the simulator.
     * @return Next path coordinates for the simulator.
     */
     Path::Trajectory CalcNext(const VehicleData &vehicleData, const Path::Trajectory &simulatorTrajectory);

  private:
    const Logic logic;
    const Predict predict;
    const int minTrajectoryLength; /**< Minimum number of coordinates to send to simulator. */
    const int maxTrajectoryLength; /**< Maximum number of coordinates to send to simulator. */

    /** Internally stored copy of previous output trajectory.
     * This contains the same x,y coordinates that also are feedback from the simulator, but it also enable storage of
     * other related data. */
    Trajectory previousTrajectory;

    /** Adjust internally stored previous trajectory to match the previous trajectory from the simulator.
     * That is, it removes already processed parts of previous trajectory. */
    void AdjustPreviousTrajectory(const Path::Trajectory &simulatorTrajectory);

    /** Calculates the cost of a trajectory. */
    double CostCalculator(const VehicleData &vehicleData, const std::vector<Path::Trajectory> &predictions,
                          const Path::Trajectory &trajectory, bool verbose);

  }; /* class Planner */
}; /* namespace Path */
#endif /* PLANNER_H */
