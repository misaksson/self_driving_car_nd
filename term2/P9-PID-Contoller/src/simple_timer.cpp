#include <chrono>
#include "simple_timer.h"

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::microseconds microseconds;

SimpleTimer::SimpleTimer() {
  previousTime_ = Clock::now();
}

SimpleTimer::~SimpleTimer() {

}

double SimpleTimer::GetDelta() {
  Clock::time_point currentTime = Clock::now();
  microseconds deltaTime = std::chrono::duration_cast<microseconds>(currentTime - previousTime_);
  previousTime_ = currentTime;

  return (double)deltaTime.count() / 1000000.0;
}
