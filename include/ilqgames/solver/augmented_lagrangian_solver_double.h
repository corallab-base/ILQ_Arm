#ifndef ILQGAMES_SOLVER_AUGMENTED_LAGRANGIAN_SOLVER_DOUBLE_H
#define ILQGAMES_SOLVER_AUGMENTED_LAGRANGIAN_SOLVER_DOUBLE_H

#include <ilqgames/dynamics/multi_player_dynamical_system_double.h>
#include <ilqgames/dynamics/multi_player_integrable_system_double.h>
#include <ilqgames/solver/problem_double.h>

#include <ilqgames/solver/game_solver_double.h>
#include <ilqgames/solver/ilq_solver_double.h>
#include <ilqgames/solver/lq_feedback_solver_double.h>
#include <ilqgames/solver/lq_open_loop_solver_double.h>
#include <ilqgames/solver/lq_solver_double.h>
#include <ilqgames/solver/solver_params_double.h>
#include <ilqgames/utils/linear_dynamics_approximation_double.h>
#include <ilqgames/utils/loop_timer.h>
#include <ilqgames/utils/operating_point_double.h>
#include <ilqgames/utils/quadratic_cost_approximation_double.h>
#include <ilqgames/utils/solver_log_double.h>
#include <ilqgames/utils/strategy_double.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <chrono>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace ilqgames {

class AugmentedLagrangianSolverDouble : public GameSolverDouble {
 public:
  ~AugmentedLagrangianSolverDouble() {}
  AugmentedLagrangianSolverDouble(const std::shared_ptr<ProblemDouble>& problem,
                            const SolverParamsDouble& params)
      : GameSolverDouble(problem, params) {
    // Modify parameters for unconstrained solver.
    SolverParamsDouble unconstrained_solver_params(params);
    unconstrained_solver_params.max_solver_iters =
        params.unconstrained_solver_max_iters;
    unconstrained_solver_.reset(
        new ILQSolverDouble(problem, unconstrained_solver_params));
  }

  // Solve this game. Returns true if converged. Defaults to 5 s runtime.
  std::shared_ptr<SolverLogDouble> Solve(bool* success = nullptr,
                                   Time max_runtime = 120.0);

 private:
  // Lower level (unconstrained) solver.
  std::unique_ptr<ILQSolverDouble> unconstrained_solver_;
};  // class AugmentedLagrangianSolverDouble

}  // namespace ilqgames

#endif