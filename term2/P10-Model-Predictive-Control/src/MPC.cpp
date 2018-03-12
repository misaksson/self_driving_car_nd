#include "MPC.h"
#include "polynomial.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"
#include <tuple>
#include <vector>

using CppAD::AD;

/** Number of timesteps. */
const size_t N = 25;
/** Timestep duration. */
const double dt = 0.05;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

const double target_speed = 22.35; //44.704; // 100 MPH

// The solver takes all the state variables and actuator
// variables in a singular vector. Thus, we should to establish
// when one variable starts and another ends to make our lifes easier.
const size_t x_start = 0;
const size_t y_start = x_start + N;
const size_t psi_start = y_start + N;
const size_t v_start = psi_start + N;
const size_t cte_start = v_start + N;
const size_t epsi_start = cte_start + N;
const size_t delta_start = epsi_start + N;
const size_t a_start = delta_start + N - 1;

// Evaluate a polynomial.
AD<double> polyeval(Eigen::VectorXd coeffs, AD<double> x) {
  AD<double> result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * CppAD::pow(x, i);
  }
  return result;
}

class FG_eval {
 public:
  /** Polynomial coefficients fitted to track waypoints. */
  const Eigen::VectorXd coeffs_;

  /** @param coeffs Polynomial coefficients fitted to track waypoints. */
  FG_eval(Eigen::VectorXd coeffs) : coeffs_(coeffs) {}

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;

  /** Operator called by ipopt solver. */
  void operator()(ADvector& fg, const ADvector& vars) {
    // The cost is stored is the first element of `fg`.
    const size_t fg_cost_idx = 0;
    const size_t fg_constraints_start = 1;
    fg[fg_cost_idx] = 0;

    // Reference state cost
    for (size_t i = 0; i < N; ++i) {
      // High factor to keep the vehicle centered on the road (for now).
      fg[fg_cost_idx] += 1000.0 * CppAD::pow(vars[cte_start + i], 2);
      // Small factor (don't care much about epsi for now).
      fg[fg_cost_idx] += 0.1 * CppAD::pow(vars[epsi_start + i], 2);
      // High factor to not make vehicle stop in curves due to the very high cost on delta.
      fg[fg_cost_idx] += 10.0 * CppAD::pow(vars[v_start + i] - target_speed, 2);
    }

    // Actuators cost
    for (size_t i = 0; i < N - 1; ++i) {
      /* Very high cost on delta to avoid bad prediction when compensating for
       * latency. The problem is that the implemented motion model doesn't seem
       * to work when the vehicle is turning sharply, which can be seen by the
       * incorrect visualization of the expected vehicle track. ToDo: This might
       * be improved by a more advanced motion model that accounts for the turn
       * rate also in the translation changes. */
      fg[fg_cost_idx] += 100000 * CppAD::pow(vars[delta_start + i], 2);
      // Small factor (don't care much about acceleration for now).
      fg[fg_cost_idx] += 1.0 * CppAD::pow(vars[a_start + i], 2);
    }

    // Change cost
    for (size_t i = 0; i < N - 2; ++i) {
      // Small factor, actuator changes doesn't seem to be a problem at this point.
      fg[fg_cost_idx] += 0.0 * CppAD::pow(vars[delta_start + i + 1] - vars[delta_start + i], 2);
      fg[fg_cost_idx] += 0.0 * CppAD::pow(vars[a_start + i + 1] - vars[a_start + i], 2);
    }

    // Should evaluate to initial state.
    fg[fg_constraints_start + x_start] = vars[x_start];
    fg[fg_constraints_start + y_start] = vars[y_start];
    fg[fg_constraints_start + psi_start] = vars[psi_start];
    fg[fg_constraints_start + v_start] = vars[v_start];
    fg[fg_constraints_start + cte_start] = vars[cte_start];
    fg[fg_constraints_start + epsi_start] = vars[epsi_start];

    for (size_t i = 1; i < N; ++i) {
      // Current time step
      const AD<double> x1 = vars[x_start + i];
      const AD<double> y1 = vars[y_start + i];
      const AD<double> psi1 = vars[psi_start + i];
      const AD<double> v1 = vars[v_start + i];
      const AD<double> cte1 = vars[cte_start + i];
      const AD<double> epsi1 = vars[epsi_start + i];

      // Previous time step
      const AD<double> x0 = vars[x_start + i - 1];
      const AD<double> y0 = vars[y_start + i - 1];
      const AD<double> psi0 = vars[psi_start + i - 1];
      const AD<double> v0 = vars[v_start + i - 1];
      const AD<double> cte0 = vars[cte_start + i - 1];
      const AD<double> epsi0 = vars[epsi_start + i - 1];
      const AD<double> delta0 = vars[delta_start + i - 1];
      const AD<double> a0 = vars[a_start + i - 1];
      const AD<double> f0 = polyeval(coeffs_, x0);
      const AD<double> psiDes0 = CppAD::atan(polyeval(Polynomial::Derivative(coeffs_), x0));

      // Should evaluate to 0 (optimally)
      fg[fg_constraints_start + x_start + i] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[fg_constraints_start + y_start + i] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[fg_constraints_start + psi_start + i] = psi1 - (psi0 + (v0 / Lf) * delta0 * dt);
      fg[fg_constraints_start + v_start + i] = v1 - (v0 + a0 * dt);
      fg[fg_constraints_start + cte_start + i] = cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
      fg[fg_constraints_start + epsi_start + i] = epsi1 - (psi0 - psiDes0 + (v0 / Lf) * delta0 * dt);
    }
  }
};


MPC::MPC(const double latency) : latency_(latency) {}

MPC::~MPC() {}

tuple<MPC::Actuations, std::vector<double>, std::vector<double>> MPC::Solve(const Eigen::VectorXd state,
                                                                           const Eigen::VectorXd coeffs) {
  typedef CPPAD_TESTVECTOR(double) Dvector;

  // The number of actuation steps in the model.
  const size_t nActuationSteps = N - 1;
  // The number of model variables (includes both states and actuations).
  const size_t nModelVars = N * state.size() + nActuationSteps * 2;
  // The number of constraints
  const size_t nModelConstraints = N * state.size();

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector modelVars(nModelVars);
  for (size_t i = 0; i < nModelVars; ++i) {
    modelVars[i] = 0;
  }

  // Set the initial variable values
  modelVars[x_start] = state[StateIdx::x];
  modelVars[y_start] = state[StateIdx::y];
  modelVars[psi_start] = state[StateIdx::psi];
  modelVars[v_start] = state[StateIdx::v];
  modelVars[cte_start] = state[StateIdx::cte];
  modelVars[epsi_start] = state[StateIdx::epsi];

  Dvector modelVarsLowerBound(nModelVars);
  Dvector modelVarsUpperBound(nModelVars);
  // Set all non-actuators upper and lowerlimits
  // to the max negative and positive values.
  for (size_t i = 0; i < delta_start; ++i) {
    modelVarsLowerBound[i] = -1.0e19;
    modelVarsUpperBound[i] = 1.0e19;
  }

  for (size_t i = 0; i < nActuationSteps; ++i) {
    // Delta upper and lower limits are set to -25 and 25 degrees (in radians).
    modelVarsLowerBound[delta_start + i] = -0.436332;
    modelVarsUpperBound[delta_start + i] = 0.436332;
    // Acceleration/decceleration upper and lower limits.
    modelVarsLowerBound[a_start + i] = -1.0;
    modelVarsUpperBound[a_start + i] = 1.0;
  }

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector modelConstraintsLowerBound(nModelConstraints);
  Dvector modelConstraintsUpperBound(nModelConstraints);
  for (size_t i = 0; i < nModelConstraints; ++i) {
    modelConstraintsLowerBound[i] = 0.0;
    modelConstraintsUpperBound[i] = 0.0;
  }

  modelConstraintsLowerBound[x_start] = state[StateIdx::x];
  modelConstraintsLowerBound[y_start] = state[StateIdx::y];
  modelConstraintsLowerBound[psi_start] = state[StateIdx::psi];
  modelConstraintsLowerBound[v_start] = state[StateIdx::v];
  modelConstraintsLowerBound[cte_start] = state[StateIdx::cte];
  modelConstraintsLowerBound[epsi_start] = state[StateIdx::epsi];

  modelConstraintsUpperBound[x_start] = state[StateIdx::x];
  modelConstraintsUpperBound[y_start] = state[StateIdx::y];
  modelConstraintsUpperBound[psi_start] = state[StateIdx::psi];
  modelConstraintsUpperBound[v_start] = state[StateIdx::v];
  modelConstraintsUpperBound[cte_start] = state[StateIdx::cte];
  modelConstraintsUpperBound[epsi_start] = state[StateIdx::epsi];

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  // options for IPOPT solver
  std::string options;
  options += "Integer print_level  0\n";
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  options += "Numeric max_cpu_time          0.5\n";

  // solve the problem
  CppAD::ipopt::solve_result<Dvector> solution;
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, modelVars, modelVarsLowerBound, modelVarsUpperBound, modelConstraintsLowerBound,
      modelConstraintsUpperBound, fg_eval, solution);

  // Check some of the solution values
  if (solution.status != CppAD::ipopt::solve_result<Dvector>::success) {
    cout << "Ipopt solution not OK" << endl;
  }

  // Provide the predicted trajectory for display.
  vector<double> predictedX, predictedY;
  for (size_t i = 0; i < N; ++i) {
    predictedX.push_back(solution.x[x_start + i]);
    predictedY.push_back(solution.x[y_start + i]);
  }

  const Actuations actuations = {
    .delta = solution.x[delta_start],
    .a = solution.x[a_start]
  };
  return make_tuple(actuations, predictedX, predictedY);
}

Eigen::VectorXd MPC::Predict(const Eigen::VectorXd state, const MPC::Actuations actuations) {
  Eigen::VectorXd predicted(static_cast<int>(StateIdx::nStates));
  predicted[StateIdx::x] = state[StateIdx::x] + state[StateIdx::v] * cos(state[StateIdx::psi]) * latency_;
  predicted[StateIdx::y] = state[StateIdx::y] + state[StateIdx::v] * sin(state[StateIdx::psi]) * latency_;
  predicted[StateIdx::psi] = state[StateIdx::psi] + (state[StateIdx::v] / Lf) * actuations.delta * latency_;
  predicted[StateIdx::v] = state[StateIdx::v] + actuations.a * latency_;
  return predicted;
}
