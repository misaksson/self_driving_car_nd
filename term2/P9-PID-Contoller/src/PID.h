#ifndef PID_H
#define PID_H

class PID {

public:
  PID();
  virtual ~PID();

  /** Initialize the PID controller.
   * @param Kp Proportional coefficient
   * @param Ki Integral coefficient
   * @param Kd Derivative coefficient
   */
  virtual void Init(double Kp, double Ki, double Kd);

  /**
   * Calculate the total PID error.
   * @param cte Crosstrack error.
   * @output PID error.
   */
  virtual double CalcError(double cte);

protected:
  /** Reset internal state. */
  virtual void Reset();

  /** Accumulated crosstrack error */
  double integral_cte_;

  /** Proportional coefficient */
  double Kp_;

  /** Integral coefficient */
  double Ki_;

  /** Derivative coefficient */
  double Kd_;

private:
  /** Previous crosstrack error */
  double previous_cte_;

};

#endif /* PID_H */
