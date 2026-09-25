#ifndef ILQGAMES_SOLVER_GAME_SOLVER_DOUBLE_H
#define ILQGAMES_SOLVER_GAME_SOLVER_DOUBLE_H

#include <ilqgames/dynamics/multi_player_dynamical_system_double.h>
#include <ilqgames/dynamics/multi_player_integrable_system_double.h>
#include <ilqgames/utils/linear_dynamics_approximation_double.h>
#include <ilqgames/utils/operating_point_double.h>
#include <ilqgames/utils/quadratic_cost_approximation_double.h>
#include <ilqgames/utils/strategy_double.h>
#include <ilqgames/utils/solver_log_double.h>

#include <ilqgames/solver/problem_double.h>
#include <ilqgames/solver/solver_params_double.h>
#include <ilqgames/solver/lq_feedback_solver_double.h>
#include <ilqgames/solver/lq_open_loop_solver_double.h>
#include <ilqgames/solver/lq_solver_double.h>

#include <ilqgames/utils/loop_timer.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <chrono>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace ilqgames {

namespace {

// Maximum number of loop times to store in loop timer.
static constexpr size_t kMaxLoopTimesToRecord = 10;

}  // anonymous namespace

class GameSolverDouble {
 public:
  virtual ~GameSolverDouble() {}

  // Solve this game. Returns true if converged.
  virtual std::shared_ptr<SolverLogDouble> Solve(
      bool* success = nullptr, Time max_runtime = constants::kInfinity) = 0;

  // Accessors.
  ProblemDouble& GetProblem() { return *problem_; }

 protected:
  GameSolverDouble(const std::shared_ptr<ProblemDouble>& problem,
             const SolverParamsDouble& params)
      : problem_(problem), params_(params), timer_(kMaxLoopTimesToRecord) {
    CHECK_NOTNULL(problem_.get());
    CHECK_NOTNULL(problem_->Dynamics().get());
  }

  // Create a new log. This may be overridden by derived classes (e.g., to
  // change the name of the log).
  virtual std::shared_ptr<SolverLogDouble> CreateNewLog() const {
    return std::make_shared<SolverLogDouble>();
  }

  // Store the underlying problem.
  const std::shared_ptr<ProblemDouble> problem_;

  // Solver parameters.
  const SolverParamsDouble params_;

  // Timer to keep track of loop execution times.
  LoopTimer timer_;
};  // class GameSolverDouble

}  // namespace ilqgames

#endif
