#ifndef MPC_H
#define MPC_H

#include <deque>
#include <tuple>
#include <vector>
#include "Eigen-3.3/Eigen/Core"

using namespace std;

class MPC {
public:
   /** @param latency The expected latency for actuations. */
  MPC(const double latency);

  virtual ~MPC();

  /** Actuations applied in the system. */
  struct Actuations {
    double delta; /**< Steering angle in radians. */
    double a;     /**< Acceleration in meter per second. */
  };
  const size_t nActuations = 2;

  struct State {
    double x;     /**< Vehicle x position. */
    double y;     /**< Vehicle y position. */
    double psi;   /**< Vehicle psi angle. */
    double v;     /**< Vehicle velocity. */
    double cte;   /**< Vehicle cross-track error. */
    double epsi;  /**< Vehicle orientation error. */
  };
  const size_t nStates = 6;

  /** Solve the model given the vehicle state vector and target path.
   * @param state State vector of the vehicle, x, y, phi, v, cte, epsi
   * @param coeffs Polynomial describing the target path.
   * @output Actuations to be applied on the system. The predicted path of the
   *         vehicle is also provided for debugging.
   */
  std::tuple<Actuations, std::vector<double>, std::vector<double>> Solve(const State state, const Eigen::VectorXd coeffs);

  /** Predict the future state to compensate for latency.
   * @param state Current state.
   * @param actuations The actuations applied during the latency.
   * @output predicted state. */
  State Predict(const State current, const Actuations actuations);

private:
  /** The expected latency for actuations. */
  const double latency_;
};

#endif /* MPC_H */
