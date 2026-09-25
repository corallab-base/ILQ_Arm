#ifndef ILQGAMES_SOLVER_LQ_SOLVER_DOUBLE_H
#define ILQGAMES_SOLVER_LQ_SOLVER_DOUBLE_H

#include <ilqgames/dynamics/multi_player_integrable_system_double.h>
#include <ilqgames/utils/linear_dynamics_approximation_double.h>
#include <ilqgames/utils/quadratic_cost_approximation_double.h>
#include <ilqgames/utils/strategy_double.h>

#include <glog/logging.h>
#include <vector>

namespace ilqgames {

class LQSolverDouble {
 public:
  virtual ~LQSolverDouble() {}

  // Solve underlying LQ game to a Nash equilibrium. This will differ in derived
  // classes depending on the information structure of the game.
  // Optionally return delta xs and costates.
  virtual std::vector<StrategyDouble> Solve(
      const std::vector<LinearDynamicsApproximationDouble>& linearization,
      const std::vector<std::vector<QuadraticCostApproximationDouble>>&
          quadraticization,
      const VectorXd& x0, std::vector<VectorXd>* delta_xs = nullptr,
      std::vector<std::vector<VectorXd>>* costates = nullptr) = 0;

 protected:
  LQSolverDouble(const std::shared_ptr<const MultiPlayerIntegrableSystemDouble>& dynamics,
           size_t num_time_steps)
      : dynamics_(dynamics), num_time_steps_(num_time_steps) {
    CHECK_NOTNULL(dynamics.get());
  }

  // Dynamics and number of time steps.
  const std::shared_ptr<const MultiPlayerIntegrableSystemDouble> dynamics_;
  const size_t num_time_steps_;
};  // class LQSolverDouble

}  // namespace ilqgames

#endif
