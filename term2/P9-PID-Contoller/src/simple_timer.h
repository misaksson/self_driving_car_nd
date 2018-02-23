#ifndef SIMPLE_TIMER_H
#define SIMPLE_TIMER_H

#include <chrono>

/** Wraps the chrono timer into a simpler interface. */
class SimpleTimer {

public:
  SimpleTimer();
  virtual ~SimpleTimer();

  /** Get time since last call of this method (or class initialization).
   * @output Time in seconds.
   */
  double GetDelta();

private:
  std::chrono::high_resolution_clock::time_point previousTime_;
};

#endif /* SIMPLE_TIMER_H */
