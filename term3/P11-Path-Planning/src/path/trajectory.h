#ifndef TRAJECTORY_H
#define TRAJECTORY_H

#include <string>
#include <vector>
#include "../helpers.h"
#include "../vehicle_data.h"

namespace Path {
  class Trajectory {
  public:
    Trajectory();
    Trajectory(int nCoords);
    Trajectory(std::vector<double> x, std::vector<double> y);
    std::vector<double> x;
    std::vector<double> y;

    /** Concatenate two trajectories. */
    Trajectory operator+(const Trajectory &other) const {
      Trajectory result = *this;
      result.x.insert(result.x.end(), other.x.begin(), other.x.end());
      result.y.insert(result.y.end(), other.y.begin(), other.y.end());
      return result;
    }

    /** Concatenate a trajectory to this trajectory. */
    void operator+=(const Trajectory &other) {
      x.insert(x.end(), other.x.begin(), other.x.end());
      y.insert(y.end(), other.y.begin(), other.y.end());
    }

    /** Output stream operator providing all coordinates. */
    friend std::ostream& operator<<(std::ostream &os, const Trajectory &m) {
      os.precision(10);
      os << std::fixed;
      for (int i = 0; i < m.size(); ++i) {
        os << "(" << m.x[i] << ", " << m.y[i] << "), ";
      }
      return os;
    }

    /** Provides the number of coordinates in the trajectory. */
    int size() const;

    /** Get an approximation of vehicle state at the end of the trajectory.
     * This is useful when concatenating trajectories. The result seems to
     * be smooth enough for the simulator.
     */
    VehicleData::EgoVehicleData getEndState(const Helpers &helpers) const;

    /** Trajectory kinematics estimations. */
    class Kinematics {
    public:
      std::vector<double> speeds;
      std::vector<double> accelerations;
      std::vector<double> jerks;
      std::vector<double> yaws;
      std::vector<double> yawRates;

      /** Output stream operator providing all kinematics for the trajectory. */
      friend std::ostream& operator<<(std::ostream &os, const Kinematics &m) {
        size_t i = 0u;
        for (; i < m.jerks.size(); ++i) {
          os << i <<
                ", speed = " << m.speeds[i] <<
                ", yaw = " << m.yaws[i] <<
                ", acc = " << m.accelerations[i] <<
                ", yawRate = " << m.yawRates[i] <<
                ", jerk = " << m.jerks[i] <<
                  std::endl;
        }
        for (; i < m.accelerations.size(); ++i) {
          os << i <<
                ", speed = " << m.speeds[i] <<
                ", yaw = " << m.yaws[i] <<
                ", acc = " << m.accelerations[i] <<
                ", yawRate = " << m.yawRates[i] <<
                  std::endl;
        }
        for (; i < m.speeds.size(); ++i) {
          os << i <<
                ", speed = " << m.speeds[i] <<
                ", yaw = " << m.yaws[i] <<
                  std::endl;
        }
        return os;
      };
    };

    /** Provides kinematics values for the trajectory. */
    Kinematics getKinematics() const;
  };

  class TrajectoryCalculator {
  public:
    TrajectoryCalculator(const Helpers &helpers);
    virtual ~TrajectoryCalculator();

    /** Apply optimal acceleration to reach delta_speed.
     * The d coordinate is kept constant. */
    Trajectory Accelerate(const VehicleData::EgoVehicleData &start, double delta_speed) const;
    /** Extend vector by a number of coordinates.
     * The speed and d coordinate is kept constant. */
    Trajectory ConstantSpeed(const VehicleData::EgoVehicleData &start, int numCoords) const;
    /** Smoothly transition from position A to B with constant acceleration.
     * The yaw angle will be the same as the road at the end of the trajectory. */
    Trajectory AdjustSpeed(const VehicleData::EgoVehicleData &start, double delta_s, double delta_d, double delta_speed) const;

   private:
     const Helpers &helpers;
  }; /* class TrajectoryCalculator */
}; /* namespace Path */

#endif /* TRAJECTORY_H */
