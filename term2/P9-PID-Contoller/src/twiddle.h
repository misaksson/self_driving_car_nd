#ifndef TWIDDLE_H
#define TWIDDLE_H

#include <string>
#include "PID.h"

class Twiddle : public PID {
public:
  Twiddle();
  Twiddle(std::string name);
  Twiddle(bool consoleOutput);

  virtual ~Twiddle();

  /** Initialize the PID controller and twiddle.
   * The twiddle algorithm will however be inactive.
   * @param Kp Proportional coefficient
   * @param Ki Integral coefficient
   * @param Kd Derivative coefficient
   */
  void Init(double Kp, double Ki, double Kd);

  /** Initialize the PID controller and twiddle.
   * @param Kp Proportional coefficient
   * @param Ki Integral coefficient
   * @param Kd Derivative coefficient
   * @param active Set to true to activate the twiddle algorithm.
   * @param name Name of instance used for debug purpose.
   * @param consoleOutput Outputs debug info to stout if true.
   */
  void Init(double Kp, double Ki, double Kd,
            bool active, std::string name, bool consoleOutput);

  /** Initialize the PID controller and twiddle.
   * @param Kp Proportional coefficient
   * @param Ki Integral coefficient
   * @param Kd Derivative coefficient
   * @param active Set to true to activate the twiddle algorithm.
   * @param name Name of instance used for debug purpose.
   * @param consoleOutput Outputs debug info to stout if true.
   */
  void Init(double Kp, double Ki, double Kd,
            double dKp, double dKi, double dKd,
            bool active, std::string name, bool consoleOutput);

  /** Reset internal states. */
  void Reset();

  /** Extends this method in the PID controller by also accumulating the error.
   * @param cte Crosstrack error.
   * @output PID error.
   */
  double CalcError(double cte);

  /** Get accumulated error. */
  double GetAccumulatedError();

  /** Update the PID controller with next parameters to try out.
   * Note, only the internal accumulated error is used for parameter
   * evaluation when using this method.
   */
  void SetNextParams();

  /** Update the PID controller with next parameters to try out.
   * @param totalError Accumulated error from all relevant controllers,
   *                   including this instance.
   */
  void SetNextParams(double totalError);

  /** Aborts any ongoing tuning and reset to best parameters. */
  void Abort();

  /** Continue tuning (from where it was aborted). */
  void Continue();

private:

  /** Name identifying this instance. */
  std::string name_;

  /** Set to false to avoid tuning result in the console. */
  bool consoleOutput_;

  /** This class behaves just as the PID class when this is false. */
  bool active_;

  /** Delta change for the proportional coefficient. */
  double dKp_;

  /** Delta change for the integral coefficient. */
  double dKi_;

  /** Delta change for the derivative coefficient. */
  double dKd_;

  /** The coefficient index currently being tuned. */
  int currentCoefficient_;

  /** The accumulated error for this run. */
  double accumulatedError_;

  /** The lowest accumulated error so far. */
  double lowestError_;

  /** Keeps track of number of iteration just to  give a hint about the progress. */
  int iteration_;

  /** Internal state for the twiddle algorithm. */
  enum NextTuningState {
    INIT,     /**< Increase first coefficient without updating any tuning parameter. */
    INCREASE, /**< Increase next coefficient. */
    DECREASE, /**< Unless improvement in current attempt, decrease current coefficient. */
    REVERT,   /**< Unless improvement in current attempt, revert current coefficient and
                   set the tuning less aggressive next time. Then continue by increasing
                   the next coefficient. */
  };

  NextTuningState nextTuning_;
};

#endif /* TWIDDLE_H */
