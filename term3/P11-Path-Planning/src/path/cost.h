#ifndef COST_H
#define COST_H

#include "../vehicle_data.h"
#include "trajectory.h"
#include <vector>
#include <typeinfo>
#include <cxxabi.h>

namespace Path {
  class Cost {
  public:
    Cost() {};
    virtual ~Cost() {};

    /** Set to true to get information about the calculated cost to console. */
    static bool verbose;

    /* Common data for all trajectories. */
    static VehicleData vehicleData;
    static Path::Trajectory previousTrajectory;
    static std::vector<Path::Trajectory> predictions;
    static VehicleData::EgoVehicleData previousEgoEndState;
    static std::vector<VehicleData::EgoVehicleData> previousOthersEndState;
    static std::vector<VehicleData::EgoVehicleData> othersEndState;
    static std::vector<std::vector<VehicleData::EgoVehicleData>> othersStateSamples;
    static std::vector<Path::Trajectory::Kinematics> othersKinematics;

    /* Current trajectory data. */
    static VehicleData::EgoVehicleData egoEndState;
    static Path::Trajectory::Kinematics egoKinematics;
    static std::vector<VehicleData::EgoVehicleData> egoStateSamples;
    static int startLane;
    static int endLane;
    static double shortestDistanceToOthers;
    static double shortestDistanceToOthersAhead;

    static void preprocessCommonData(const Path::Trajectory previousTrajectory_, const VehicleData vehicleData_, const std::vector<Path::Trajectory> predictions_);
    static void preprocessCurrentTrajectory(const Path::Trajectory &trajectory);

    double getCost(const Path::Trajectory &trajectory) {
      const double cost = calc(trajectory);
      if (verbose) {
        int status;
        char * demangled = abi::__cxa_demangle(typeid(*this).name(), 0, 0, &status);
        std::cout << demangled << " cost = " << cost << std::endl;
        free(demangled);
      }
      return cost;
    }
  protected:
    virtual double calc(const Path::Trajectory &trajectory) const = 0;

  };

  class SlowSpeed : public Cost {
  public:
    const double slowSpeedCostFactor;
    SlowSpeed(double slowSpeedCostFactor) : slowSpeedCostFactor(slowSpeedCostFactor) {};
    virtual ~SlowSpeed() {};
  private:
    double calc(const Path::Trajectory &trajectory) const;
  };

  class ExceedSpeedLimit : public Cost {
  public:
    const double exceedSpeedLimitCost;
    ExceedSpeedLimit(double exceedSpeedLimitCost) : exceedSpeedLimitCost(exceedSpeedLimitCost) {};
    virtual ~ExceedSpeedLimit() {};
  private:
    double calc(const Path::Trajectory &trajectory) const;
  };
  class ChangeIntention : public Cost {
  public:
    const double changeIntentionCost;
    ChangeIntention(double changeIntentionCost) : changeIntentionCost(changeIntentionCost) {};
    virtual ~ChangeIntention() {};
  private:
    double calc(const Path::Trajectory &trajectory) const;
  };

  class LaneChange : public Cost {
  public:
    const double laneChangeCostFactor;
    LaneChange(double laneChangeCostFactor) : laneChangeCostFactor(laneChangeCostFactor) {};
    virtual ~LaneChange() {};
  private:
    double calc(const Path::Trajectory &trajectory) const;
  };

  class LaneChangeInFrontOfOther : public Cost {
  public:
    const double laneChangeInFrontOfOtherCost;
    LaneChangeInFrontOfOther(double laneChangeInFrontOfOtherCost) : laneChangeInFrontOfOtherCost(laneChangeInFrontOfOtherCost) {};
    virtual ~LaneChangeInFrontOfOther() {};
  private:
    double calc(const Path::Trajectory &trajectory) const;
  };

  class LaneChangeInFrontOfOtherFaster : public Cost {
  public:
    const double laneChangeInFrontOfOtherFasterCost;
    LaneChangeInFrontOfOtherFaster(double laneChangeInFrontOfOtherFasterCost) : laneChangeInFrontOfOtherFasterCost(laneChangeInFrontOfOtherFasterCost) {};
    virtual ~LaneChangeInFrontOfOtherFaster() {};
  private:
    double calc(const Path::Trajectory &trajectory) const;
  };

  class NearOtherVehicles : public Cost {
  public:
    const double inverseDistanceCostFactor;
    NearOtherVehicles(double inverseDistanceCostFactor) : inverseDistanceCostFactor(inverseDistanceCostFactor) {};
    virtual ~NearOtherVehicles() {};
  private:
    double calc(const Path::Trajectory &trajectory) const;
  };

  class Collision : public Cost {
  public:
    const double collisionDistance;
    const double collisionCost;
    Collision(double collisionCost) : collisionCost(collisionCost), collisionDistance(3.0) {};
    virtual ~Collision() {};
  private:
    double calc(const Path::Trajectory &trajectory) const;
  };

  class SlowLane : public Cost {
  public:
    const double slowLaneCostFactor;
    SlowLane(double slowLaneCostFactor) : slowLaneCostFactor(slowLaneCostFactor) {};
    virtual ~SlowLane() {};
  private:
    double calc(const Path::Trajectory &trajectory) const;
  };

  class ViolateCriticalDistanceAhead : public Cost {
  public:
    const double acceptedLongitudinalTimeDiff;
    const double criticalLongitudinalTimeDiff;
    const double violateCriticalLongitudinalTimeDiffCost;
    ViolateCriticalDistanceAhead(double violateCriticalLongitudinalTimeDiffCost) :
        violateCriticalLongitudinalTimeDiffCost(violateCriticalLongitudinalTimeDiffCost),
        acceptedLongitudinalTimeDiff(0.7), criticalLongitudinalTimeDiff(0.3) {};
    virtual ~ViolateCriticalDistanceAhead() {};
  private:
    double calc(const Path::Trajectory &trajectory) const;
  };

  class Acceleration : public Cost {
  public:
    const double accelerationCostFactor;
    Acceleration(double accelerationCostFactor) : accelerationCostFactor(accelerationCostFactor) {};
    virtual ~Acceleration() {};
  private:
    double calc(const Path::Trajectory &trajectory) const;
  };

  class Jerk : public Cost {
  public:
    const double jerkCostFactor;
    Jerk(double jerkCostFactor) : jerkCostFactor(jerkCostFactor) {};
    virtual ~Jerk() {};
  private:
    double calc(const Path::Trajectory &trajectory) const;
  };

  class YawRate : public Cost {
  public:
    const double yawRateCostFactor;
    YawRate(double yawRateCostFactor) : yawRateCostFactor(yawRateCostFactor) {};
    virtual ~YawRate() {};
  private:
    double calc(const Path::Trajectory &trajectory) const;
  };

  class ExceedAccelerationLimit : public Cost {
  public:
    const double exceedAccelerationLimitCost;
    ExceedAccelerationLimit(double exceedAccelerationLimitCost) : exceedAccelerationLimitCost(exceedAccelerationLimitCost) {};
    virtual ~ExceedAccelerationLimit() {};
  private:
    double calc(const Path::Trajectory &trajectory) const;
  };

};

#endif /* COST_H */
