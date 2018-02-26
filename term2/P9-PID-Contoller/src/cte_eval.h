#ifndef CTE_EVAL_H
#define CTE_EVAL_H

#include <array>
#include <cmath>
#include <deque>
#include <tuple>

class CrosstrackErrorEvaluator {
public:
  CrosstrackErrorEvaluator(bool consoleOutput);
  ~CrosstrackErrorEvaluator();

  enum Performance {
    DEFECTIVE = 0,
    BAD,
    RISKY,
    OK,
    GOOD,
    IDEAL,

    NUM_PERFORMANCE_LEVELS,
  };

  Performance Evaluate(double deltaTime, double cte);

private:
  /** Set to false to avoid debug output in the console. */
  bool consoleOutput_;

  /** Current state of this algorithm. */
  Performance currentPerformance_;

  /** Parameters defining a performance level. */
  struct PerformanceLevel {
    /** Name of performance level used for debug purpose. */
    std::string name;
    /** Integral CTE per second must increase above this threshold to trigger this performance level. */
    double cteSqUpperTh_;
    /** Integral CTE per second must decrease below this threshold to release this performance level. */
    double cteSqLowerTh_;
    /** Integral CTE per second must increase above this threshold to trigger this performance level. */
    double deltaCteSqUpperTh_;
    /** Integral CTE per second must decrease below this threshold to release this performance level. */
    double deltaCteSqLowerTh_;
    /** The number of seconds to integrate CTE when triggering this performance level. */
    double triggerPeriod_;
    /** The number of seconds to integrate CTE when releasing this performance level. */
    double releasePeriod_;
  };

  const std::array<PerformanceLevel, NUM_PERFORMANCE_LEVELS> performanceLevels_ = {{
    {
      .name = "DEFECTIVE",
      .cteSqUpperTh_ = 0.5, .cteSqLowerTh_ = 0.2,
      .deltaCteSqUpperTh_ = 0.15, .deltaCteSqLowerTh_ = 0.001,
      .triggerPeriod_ = 0.12, .releasePeriod_ = 0.3
    }, {
      .name = "BAD",
      .cteSqUpperTh_ = 0.5, .cteSqLowerTh_ = 0.2,
      .deltaCteSqUpperTh_ = 0.15, .deltaCteSqLowerTh_ = 0.001,
      .triggerPeriod_ = 0.12, .releasePeriod_ = 0.3
    }, {
      .name = "RISKY",
      .cteSqUpperTh_ = 0.5, .cteSqLowerTh_ = 0.2,
      .deltaCteSqUpperTh_ = 0.15, .deltaCteSqLowerTh_ = 0.001,
      .triggerPeriod_ = 0.12, .releasePeriod_ = 0.3
    }, {
      .name = "OK",
      .cteSqUpperTh_ = 0.5, .cteSqLowerTh_ = 0.2,
      .deltaCteSqUpperTh_ = 0.15, .deltaCteSqLowerTh_ = 0.001,
      .triggerPeriod_ = 0.12, .releasePeriod_ = 0.3
    }, {
      .name = "GOOD",
      .cteSqUpperTh_ = 0.5, .cteSqLowerTh_ = 0.2,
      .deltaCteSqUpperTh_ = 0.15, .deltaCteSqLowerTh_ = 0.001,
      .triggerPeriod_ = 0.12, .releasePeriod_ = 0.3
    }, {
      .name = "IDEAL",
      .cteSqUpperTh_ = 0.5, .cteSqLowerTh_ = 0.2,
      .deltaCteSqUpperTh_ = 0.0, .deltaCteSqLowerTh_ = 0.0,
      .triggerPeriod_ = 0.0, .releasePeriod_ = 0.0
    },
  }};

  struct HistoryElement {
    double deltaTime;  /**< Time duration since previous measurement. */
    double cteSq;      /**< Crosstrack error squared. */
    double deltaCteSq; /**< Delta crosstrack error squared. */
  };

  std::deque<HistoryElement> history_;
  double previousCte_;
  double longestIntegralPeriod_;

  bool TryTrigger(const PerformanceLevel &pl);
  bool TryRelease(const PerformanceLevel &pl);
  std::tuple<double, double> Integrate(const double integralPeriod);
  void ReduceHistory();
};

#endif /* CTE_EVAL_H */
