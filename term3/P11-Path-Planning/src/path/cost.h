#ifndef COST_H
#define COST_H

#include "../vehicle_data.h"
#include "trajectory.h"
#include <vector>

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

    virtual double calc(const Path::Trajectory &trajectory) const = 0;
  };

  class SlowSpeed : public Cost {
  public:
    const double slowSpeedCostFactor;
    SlowSpeed(double slowSpeedCostFactor) : slowSpeedCostFactor(slowSpeedCostFactor) {};
    virtual ~SlowSpeed() {};
    double calc(const Path::Trajectory &trajectory) const;
  };

  class ExceedSpeedLimit : public Cost {
  public:
    const double exceedSpeedLimitCost;
    ExceedSpeedLimit(double exceedSpeedLimitCost) : exceedSpeedLimitCost(exceedSpeedLimitCost) {};
    virtual ~ExceedSpeedLimit() {};
    double calc(const Path::Trajectory &trajectory) const;
  };
  class ChangeIntention : public Cost {
  public:
    const double changeIntentionCost;
    ChangeIntention(double changeIntentionCost) : changeIntentionCost(changeIntentionCost) {};
    virtual ~ChangeIntention() {};
    double calc(const Path::Trajectory &trajectory) const;
  };

  class LaneChange : public Cost {
  public:
    const double laneChangeCostFactor;
    LaneChange(double laneChangeCostFactor) : laneChangeCostFactor(laneChangeCostFactor) {};
    virtual ~LaneChange() {};
    double calc(const Path::Trajectory &trajectory) const;
  };

  class LaneChangeInFrontOfOther : public Cost {
  public:
    const double laneChangeInFrontOfOtherCost;
    LaneChangeInFrontOfOther(double laneChangeInFrontOfOtherCost) : laneChangeInFrontOfOtherCost(laneChangeInFrontOfOtherCost) {};
    virtual ~LaneChangeInFrontOfOther() {};
    double calc(const Path::Trajectory &trajectory) const;
  };

  class NearOtherVehicles : public Cost {
  public:
    const double inverseDistanceCostFactor;
    NearOtherVehicles(double inverseDistanceCostFactor) : inverseDistanceCostFactor(inverseDistanceCostFactor) {};
    virtual ~NearOtherVehicles() {};
    double calc(const Path::Trajectory &trajectory) const;
  };

  class Collision : public Cost {
  public:
    const double collisionDistance;
    const double collisionCost;
    Collision(double collisionCost) : collisionCost(collisionCost), collisionDistance(3.0) {};
    virtual ~Collision() {};
    double calc(const Path::Trajectory &trajectory) const;
  };

  class SlowLane : public Cost {
  public:
    const double slowLaneCostFactor;
    SlowLane(double slowLaneCostFactor) : slowLaneCostFactor(slowLaneCostFactor) {};
    virtual ~SlowLane() {};
    double calc(const Path::Trajectory &trajectory) const;
  };

  class ViolateRecommendedDistanceAhead : public Cost {
  public:
    const double recommendedLongitudinalTimeDiff;
    const double violateRecommendedLongitudinalTimeDiffCost;
    ViolateRecommendedDistanceAhead(double violateRecommendedLongitudinalTimeDiffCost) :
        violateRecommendedLongitudinalTimeDiffCost(violateRecommendedLongitudinalTimeDiffCost),
        recommendedLongitudinalTimeDiff(3.0) {};
    virtual ~ViolateRecommendedDistanceAhead() {};
    double calc(const Path::Trajectory &trajectory) const;
  };

  class ViolateCriticalDistanceAhead : public Cost {
  public:
    const double criticalLongitudinalTimeDiff;
    const double violateCriticalLongitudinalTimeDiffCost;
    ViolateCriticalDistanceAhead(double violateCriticalLongitudinalTimeDiffCost) :
        violateCriticalLongitudinalTimeDiffCost(violateCriticalLongitudinalTimeDiffCost),
        criticalLongitudinalTimeDiff(2.0) {};
    virtual ~ViolateCriticalDistanceAhead() {};
    double calc(const Path::Trajectory &trajectory) const;
  };

  class Acceleration : public Cost {
  public:
    const double accelerationCostFactor;
    Acceleration(double accelerationCostFactor) : accelerationCostFactor(accelerationCostFactor) {};
    virtual ~Acceleration() {};
    double calc(const Path::Trajectory &trajectory) const;
  };

  class Jerk : public Cost {
  public:
    const double jerkCostFactor;
    Jerk(double jerkCostFactor) : jerkCostFactor(jerkCostFactor) {};
    virtual ~Jerk() {};
    double calc(const Path::Trajectory &trajectory) const;
  };

  class YawRate : public Cost {
  public:
    const double yawRateCostFactor;
    YawRate(double yawRateCostFactor) : yawRateCostFactor(yawRateCostFactor) {};
    virtual ~YawRate() {};
    double calc(const Path::Trajectory &trajectory) const;
  };

  class ExceedAccelerationLimit : public Cost {
  public:
    const double exceedAccelerationLimitCost;
    ExceedAccelerationLimit(double exceedAccelerationLimitCost) : exceedAccelerationLimitCost(exceedAccelerationLimitCost) {};
    virtual ~ExceedAccelerationLimit() {};
    double calc(const Path::Trajectory &trajectory) const;
  };

};

#endif /* COST_H */
