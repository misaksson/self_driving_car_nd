#ifndef PREDICT_H
#define PREDICT_H

#include "trajectory.h"
#include <vector>

namespace Path {
  class Predict {
  public:
    Predict() : minPredictionLength(400) {};
    virtual ~Predict() {};

    /** Predicts a trajectory for each of the other vehicles.
     * The predicted trajectories are calculated to match what's appended to previous trajectory.
     * @param otherVehicles The vehicle data to base prediction on.
     * @param numPrevious Number of time steps to skip ahead to make the prediction match appended trajectory.
     */
    std::vector<Trajectory> calc(const std::vector<VehicleData::OtherVehicleData>  &otherVehicles, size_t numPrevious) const;
  private:
    const size_t minPredictionLength;

    /** Predict a trajectory that consider possibly ongoing lane changes.
     * This is the main method used for predictions, used whenever the Frenet coordinates seems to be reasonable */
    Trajectory inFrenetSpace(const VehicleData::OtherVehicleData &otherVehicle) const;
    /** Prediction that assumes constant velocity in x and y direction.
     * This is the fallback method when the Frenet coordinates is indicated to be broken. */
    Trajectory inCartesianSpace(const VehicleData::OtherVehicleData &otherVehicle) const;
    /** Predict vehicles with zero or negative velocity. */
    Trajectory standStill(const VehicleData::OtherVehicleData &otherVehicle) const;
  };

}; /* namespace Path */

#endif /* PREDICT_H */
