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

  /** Set internal state as in provided PID controller.
   * This makes it possible to implement interchangeable PID controllers
   * that doesn't loose internal state when changing.
   */
  void SetState(const PID &pid);

  /** Get the integrated Crosstrack error. */
  double GetIntegralCte() const;

  /** Get previous Crosstrack error. */
  double GetPreviousCte() const;

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
