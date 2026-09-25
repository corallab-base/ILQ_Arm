#ifndef ILQGAMES_SOLVER_ILQ_SOLVER_DOUBLE_H
#define ILQGAMES_SOLVER_ILQ_SOLVER_DOUBLE_H

#include <ilqgames/cost/player_cost_double.h>
#include <ilqgames/dynamics/multi_player_dynamical_system_double.h>
#include <ilqgames/solver/game_solver_double.h>
#include <ilqgames/solver/lq_feedback_solver_double.h>

#include <ilqgames/solver/lq_solver_double.h>

#include <ilqgames/solver/solver_params_double.h>
#include <ilqgames/utils/linear_dynamics_approximation_double.h>
#include <ilqgames/utils/operating_point_double.h>
#include <ilqgames/utils/quadratic_cost_approximation_double.h>
#include <ilqgames/utils/solver_log_double.h>
#include <ilqgames/utils/strategy_double.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <limits>
#include <memory>
#include <vector>

namespace ilqgames {

class ILQSolverDouble : public GameSolverDouble {
public:
  virtual ~ILQSolverDouble() {}
  ILQSolverDouble(const std::shared_ptr<ProblemDouble> &problem,
            const SolverParamsDouble &params = SolverParamsDouble())
      : GameSolverDouble(problem, params), linearization_(time::kNumTimeSteps),
        cost_quadraticization_(time::kNumTimeSteps),
        last_merit_function_value_(constants::kInfinity),
        expected_decrease_(constants::kInfinity) {
    // Set up LQ solver.
    if (params_.open_loop)
      lq_solver_.reset(
          new LQOpenLoopSolverDouble(problem_->Dynamics(), time::kNumTimeSteps));
    else
      lq_solver_.reset(
          new LQFeedbackSolverDouble(problem_->Dynamics(), time::kNumTimeSteps));

    // If this system is flat then compute the linearization once, now.
    if (problem_->Dynamics()->TreatAsLinear())
      ComputeLinearization(&linearization_);

    // Prepopulate quadraticization.
    for (auto &quads : cost_quadraticization_)
      quads.resize(problem_->Dynamics()->NumPlayers(),
                   QuadraticCostApproximationDouble(problem_->Dynamics()->XDim()));

    // Set last quadraticization to current, to start.
    last_cost_quadraticization_ = cost_quadraticization_;
  }

  // Solve this game. Returns true if converged.
  virtual std::shared_ptr<SolverLogDouble>
  Solve(bool *success = nullptr,
        Time max_runtime = std::numeric_limits<Time>::infinity());

  // Accessors.
  // NOTE: these should be primarily used by higher-level solvers.
  std::vector<std::vector<QuadraticCostApproximationDouble>> *Quadraticization() {
    return &cost_quadraticization_;
  }

  std::vector<LinearDynamicsApproximationDouble> *Linearization() {
    return &linearization_;
  }

protected:
  // Modify LQ strategies to improve convergence properties.
  // This function performs an Armijo linesearch and returns true if successful.
  bool ModifyLQStrategies(const std::vector<VectorXd> &delta_xs,
                          const std::vector<std::vector<VectorXd>> &costates,
                          std::vector<StrategyDouble> *strategies,
                          OperatingPointDouble *current_operating_point,
                          bool *has_converged);

  // Compute distance (infinity norm) between states in the given dimensions.
  // If dimensions empty, checks all dimensions.
  double StateDistance(const VectorXd &x1, const VectorXd &x2,
                      const std::vector<Dimension> &dims) const;

  // Check if solver has converged.
  virtual bool HasConverged(double current_merit_function_value) const {
    return (current_merit_function_value <= last_merit_function_value_) &&
           std::abs(last_merit_function_value_ - current_merit_function_value) <
               params_.convergence_tolerance;
  }

  // Compute overall costs and set times of extreme costs.
  void TotalCosts(const OperatingPointDouble &current_op,
                  std::vector<double> *total_costs) const;

  // Armijo condition check. Returns true if the new operating point satisfies
  // the Armijo condition, and also returns current merit function value.
  bool CheckArmijoCondition(double current_merit_function_value,
                            double current_stepsize) const;

  // Compute current merit function value. Note that to compute the merit
  // function at the given operating point we have to compute a full cost
  // quadraticization there. To do so efficiently, this will overwrite the
  // current cost quadraticization (and presume it has already been used to
  // compute the expected decrease from the last iterate).
  double MeritFunction(const OperatingPointDouble &current_op,
                      const std::vector<std::vector<VectorXd>> &costates);

  // Compute expected decrease based on current cost quadraticization,
  // (player-indexed) strategies, and (time-indexed) lists of delta states and
  // (also player-indexed) costates.
  double
  ExpectedDecrease(const std::vector<StrategyDouble> &strategies,
                   const std::vector<VectorXd> &delta_xs,
                   const std::vector<std::vector<VectorXd>> &costates) const;

  // Compute the current operating point based on the current set of
  // strategies and the last operating point.
  void CurrentOperatingPoint(const OperatingPointDouble &last_operating_point,
                             const std::vector<StrategyDouble> &current_strategies,
                             OperatingPointDouble *current_operating_point) const;

  // Populate the given vector with a linearization of the dynamics about
  // the given operating point. Provide version with no operating point for use
  // with feedback linearizable systems.
  void
  ComputeLinearization(const OperatingPointDouble &op,
                       std::vector<LinearDynamicsApproximationDouble> *linearization);
  void
  ComputeLinearization(std::vector<LinearDynamicsApproximationDouble> *linearization);

  // Compute the quadratic cost approximation at the given operating point.
  void ComputeCostQuadraticization(
      const OperatingPointDouble &op,
      std::vector<std::vector<QuadraticCostApproximationDouble>> *q);

  // Linearization and quadraticization. Both are time-indexed (and
  // quadraticizations' inner vector is indexed by player). Also keep track of
  // the quadraticization from last iteration.
  std::vector<LinearDynamicsApproximationDouble> linearization_;
  std::vector<std::vector<QuadraticCostApproximationDouble>> cost_quadraticization_;
  std::vector<std::vector<QuadraticCostApproximationDouble>>
      last_cost_quadraticization_;

  // Core LQ Solver.
  std::unique_ptr<LQSolverDouble> lq_solver_;

  // Last merit function value and expected decreases (per step length).
  double last_merit_function_value_;
  double expected_decrease_;
}; // class ILQSolverDouble

} // namespace ilqgames

#endif
