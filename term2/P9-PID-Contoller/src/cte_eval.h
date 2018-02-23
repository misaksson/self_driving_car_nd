#ifndef CTE_EVAL_H
#define CTE_EVAL_H

#include <deque>

class CrosstrackErrorEvaluator {
public:
  CrosstrackErrorEvaluator();
  ~CrosstrackErrorEvaluator();

  bool IsOk(double deltaTime, double cte);

private:
  /** Current state of this algorithm. */
  bool isOk_;
  /** Integral CTE per second must increase above this threshold to qualify as not OK. */
  const double cteSqUpperTh_ = 0.5;
  /** Integral CTE per second must decrease below this threshold to qualify as OK. */
  const double cteSqLowerTh_ = 0.2;
  /** Integral CTE per second must increase above this threshold to qualify as not OK. */
  const double deltaCteSqUpperTh_ = 0.15;
  /** Integral CTE per second must decrease below this threshold to qualify as OK. */
  const double deltaCteSqLowerTh_ = 0.001;
  /** The number of seconds to integrate CTE in transition to state not OK. */
  const double triggerPeriod_ = 0.12;
  /** The number of seconds to integrate CTE in transition to state OK. */
  const double releasePeriod_ = 0.3;

  struct HistoryElement {
    double deltaTime;  /**< Time duration since previous measurement. */
    double cteSq;      /**< Crosstrack error squared. */
    double deltaCteSq; /**< Delta crosstrack error squared. */
  };

  std::deque<HistoryElement> history_;
  double previousCte_;
};

#endif /* CTE_EVAL_H */
