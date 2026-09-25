#ifndef ILQGAMES_SOLVER_PROBLEM_DOUBLE_H
#define ILQGAMES_SOLVER_PROBLEM_DOUBLE_H

#include <ilqgames/cost/player_cost_double.h>
#include <ilqgames/dynamics/multi_player_dynamical_system_double.h>
#include <ilqgames/dynamics/multi_player_flat_system_double.h>
#include <ilqgames/dynamics/multi_player_integrable_system_double.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/strategy_double.h>
#include <ilqgames/utils/types.h>

#include <limits>
#include <memory>
#include <vector>

namespace ilqgames {

class ProblemDouble {
 public:
  virtual ~ProblemDouble() {}

  // Initialize this object.
  virtual void Initialize() {
    ConstructDynamics();
    ConstructPlayerCosts();
    ConstructInitialState();
    ConstructInitialOperatingPoint();
    ConstructInitialStrategies();
    initialized_ = true;
  }

  // Reset the initial time and change nothing else.
  void ResetInitialTime(Time t0) {
    CHECK(initialized_);
    operating_point_->t0 = t0;
  }

  void ResetInitialState(const VectorXd& x0) {
    CHECK(initialized_);
    x0_ = x0;
  }

  // Update initial state and modify previous strategies and operating
  // points to start at the specified runtime after the current time t0.
  // Since time is continuous and we will want to maintain the same fixed
  // discretization, we will integrate x0 forward from t0 by approximately
  // planner_runtime, then find the nearest state in the existing plan to that
  // state, and start from there. By default, extends operating points and
  // strategies as follows:
  // 1. new controls are zero
  // 2. new states are those that result from zero control
  // 3. new strategies are also zero
  virtual void SetUpNextRecedingHorizon(const VectorXd& x0, Time t0,
                                        Time planner_runtime = 0.1);

  // Overwrite existing solution with the given operating point and strategies.
  // Truncates to fit in the same memory.
  virtual void OverwriteSolution(const OperatingPointDouble& operating_point,
                                 const std::vector<StrategyDouble>& strategies);

  // Accessors.
  bool IsConstrained() const;
  virtual Time InitialTime() const { return operating_point_->t0; }
  const VectorXd& InitialState() const { return x0_; }
  std::vector<PlayerCostDouble>& PlayerCosts() { return player_costs_; }
  const std::vector<PlayerCostDouble>& PlayerCosts() const { return player_costs_; }
  const std::shared_ptr<const MultiPlayerIntegrableSystemDouble>& Dynamics() const {
    return dynamics_;
  }
  const MultiPlayerDynamicalSystemDouble& NormalDynamics() const {
    CHECK(!dynamics_->TreatAsLinear());
    return *static_cast<const MultiPlayerDynamicalSystemDouble*>(dynamics_.get());
  }
  const MultiPlayerFlatSystemDouble& FlatDynamics() const {
    CHECK(dynamics_->TreatAsLinear());
    return *static_cast<const MultiPlayerFlatSystemDouble*>(dynamics_.get());
  }
  virtual const OperatingPointDouble& CurrentOperatingPoint() const {
    return *operating_point_;
  }
  virtual const std::vector<StrategyDouble>& CurrentStrategies() const {
    return *strategies_;
  }

  virtual std::vector<double> Qs(const VectorXd& x) const = 0;

 protected:
  ProblemDouble();

  // Functions for initialization. By default, operating point and strategies
  // are initialized to zero.
  virtual void ConstructDynamics() = 0;
  virtual void ConstructPlayerCosts() = 0;
  virtual void ConstructInitialState() = 0;
  virtual void ConstructInitialOperatingPoint() {
    operating_point_.reset(
        new OperatingPointDouble(time::kNumTimeSteps, 0.0, dynamics_));
  }
  virtual void ConstructInitialStrategies() {
    strategies_.reset(new std::vector<StrategyDouble>());
    for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++)
      strategies_->emplace_back(time::kNumTimeSteps, dynamics_->XDim(),
                                dynamics_->UDim(ii));
  }

  // Utility used by SetUpNextRecedingHorizon. Integrate the given state
  // forward, set the new initial state and time, and return the first timestep
  // in the new problem.
  size_t SyncToExistingProblem(const VectorXd& x0, Time t0,
                               Time planner_runtime, OperatingPointDouble& op);

  // Dynamical system.
  std::shared_ptr<const MultiPlayerIntegrableSystemDouble> dynamics_;

  // Player costs. These will not change during operation of this solver.
  std::vector<PlayerCostDouble> player_costs_;

  // Initial condition.
  VectorXd x0_;

  // Strategies and operating points for all players.
  std::unique_ptr<OperatingPointDouble> operating_point_;
  std::unique_ptr<std::vector<StrategyDouble>> strategies_;

  // Has this object been initialized?
  bool initialized_;
};  // class ProblemDouble

}  // namespace ilqgames

#endif
