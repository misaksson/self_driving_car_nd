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

const double target_speed = 40.0;

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
      fg[fg_cost_idx] += CppAD::pow(vars[cte_start + i], 2);
      fg[fg_cost_idx] += CppAD::pow(vars[epsi_start + i], 2);
      fg[fg_cost_idx] += CppAD::pow(vars[v_start + i] - target_speed, 2);
    }

    // Actuators cost
    for (size_t i = 0; i < N - 1; ++i) {
      fg[fg_cost_idx] += CppAD::pow(vars[delta_start + i], 2);
      fg[fg_cost_idx] += CppAD::pow(vars[a_start + i], 2);
    }

    // Change cost
    for (size_t i = 0; i < N - 2; ++i) {
      fg[fg_cost_idx] += CppAD::pow(vars[delta_start + i + 1] - vars[delta_start + i], 2);
      fg[fg_cost_idx] += CppAD::pow(vars[a_start + i + 1] - vars[a_start + i], 2);
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

      // Should evaluate to 0
      fg[fg_constraints_start + x_start + i] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[fg_constraints_start + y_start + i] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[fg_constraints_start + psi_start + i] = psi1 - (psi0 + (v0 / Lf) * delta0 * dt);
      fg[fg_constraints_start + v_start + i] = v1 - (v0 + a0 * dt);
      fg[fg_constraints_start + cte_start + i] = cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
      fg[fg_constraints_start + epsi_start + i] = epsi1 - (psi0 - psiDes0 + (v0 / Lf) * delta0 * dt);
    }
  }
};


MPC::MPC() {}
MPC::~MPC() {}

tuple<double, double, std::vector<double>, std::vector<double>> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  typedef CPPAD_TESTVECTOR(double) Dvector;

  const double x = state[0];
  const double y = state[1];
  const double psi = state[2];
  const double v = state[3];
  const double cte = state[4];
  const double epsi = state[5];

  // The number of model variables (includes both states and inputs).
  const size_t n_vars = N * state.size() + (N - 1) * 2;
  // The number of constraints
  const size_t n_constraints = N * state.size();

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (size_t i = 0; i < n_vars; ++i) {
    vars[i] = 0;
  }

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
  // Set all non-actuators upper and lowerlimits
  // to the max negative and positive values.
  for (size_t i = 0; i < delta_start; ++i) {
    vars_lowerbound[i] = -1.0e19;
    vars_upperbound[i] = 1.0e19;
  }

  // The upper and lower limits of delta are set to -25 and 25
  // degrees (values in radians).
  for (size_t i = delta_start; i < a_start; ++i) {
    vars_lowerbound[i] = -0.436332;
    vars_upperbound[i] = 0.436332;
  }

  // Acceleration/decceleration upper and lower limits.
  for (size_t i = a_start; i < n_vars; ++i) {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] = 1.0;
  }

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (size_t i = 0; i < n_constraints; ++i) {
    constraints_lowerbound[i] = 0.0;
    constraints_upperbound[i] = 0.0;
  }

  constraints_lowerbound[x_start] = x;
  constraints_lowerbound[y_start] = y;
  constraints_lowerbound[psi_start] = psi;
  constraints_lowerbound[v_start] = v;
  constraints_lowerbound[cte_start] = cte;
  constraints_lowerbound[epsi_start] = epsi;

  constraints_upperbound[x_start] = x;
  constraints_upperbound[y_start] = y;
  constraints_upperbound[psi_start] = psi;
  constraints_upperbound[v_start] = v;
  constraints_upperbound[cte_start] = cte;
  constraints_upperbound[epsi_start] = epsi;

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
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

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

  return make_tuple(-solution.x[delta_start], solution.x[a_start], predictedX, predictedY);
}
