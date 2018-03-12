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

  /** State vector indexes. */
  enum StateIdx {
    x = 0,  /**< State vector index of vehicle x position. */
    y,      /**< State vector index of vehicle y position. */
    psi,    /**< State vector index of vehicle psi angle. */
    v,      /**< State vector index of vehicle velocity. */
    cte,    /**< State vector index of vehicle cross-track error. */
    epsi,   /**< State vector index of vehicle orientation error. */

    /** Number of state variables. */
    nStates
  };

  /** Solve the model given the vehicle state vector and target path.
   * @param state State vector of the vehicle, x, y, phi, v, cte, epsi
   * @param coeffs Polynomial describing the target path.
   * @output Actuations to be applied on the system. The predicted path of the
   *         vehicle is also provided for debugging.
   */
  std::tuple<Actuations, std::vector<double>, std::vector<double>> Solve(const Eigen::VectorXd state, const Eigen::VectorXd coeffs);

  /** Predict the future state to compensate for latency.
   * @param state Current state.
   * @param actuations The actuations applied during the latency.
   * @output predicted future state. */
  Eigen::VectorXd Predict(const Eigen::VectorXd state, const Actuations actuations);

private:
  /** The expected latency for actuations. */
  const double latency_;
};

#endif /* MPC_H */
