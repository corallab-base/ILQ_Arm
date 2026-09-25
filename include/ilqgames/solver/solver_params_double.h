#ifndef ILQGAMES_SOLVER_SOLVER_PARAMS_DOUBLE_H
#define ILQGAMES_SOLVER_SOLVER_PARAMS_DOUBLE_H

#include <ilqgames/utils/types.h>

namespace ilqgames {

struct SolverParamsDouble {
  // Consider a solution converged once max elementwise difference is below this
  // tolerance or solver has exceeded a maximum number of iterations.
  double convergence_tolerance = 1e-1;
  size_t max_solver_iters = 1000;

  // Linesearch parameters. If flag is set 'true', then applied initial alpha
  // scaling to all strategies and backs off geometrically at the given rate for
  // the specified number of steps.
  bool linesearch = true;
  double initial_alpha_scaling = 0.5;
  double geometric_alpha_scaling = 0.5;
  size_t max_backtracking_steps = 10;
  double expected_decrease_fraction = 0.1;

  // Whether solver should shoot for an open loop or feedback Nash.
  bool open_loop = false;

  // State and control regularization.
  double state_regularization = 0.0;
  double control_regularization = 0.0;

  // Augmented Lagrangian parameters.
  size_t unconstrained_solver_max_iters = 10;
  double geometric_mu_scaling = 1.1;
  double geometric_mu_downscaling = 0.5;
  double geometric_lambda_downscaling = 0.5;
  double constraint_error_tolerance = 1e-1;

  // Should the solver reset problem/constraint params to their initial values.
  // NOTE: defaults to true.
  bool reset_problem = true;
  bool reset_lambdas = true;
  bool reset_mu = true;
};  // struct SolverParamsDouble

}  // namespace ilqgames

#endif
