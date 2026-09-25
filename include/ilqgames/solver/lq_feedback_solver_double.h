#ifndef ILQGAMES_SOLVER_LQ_FEEDBACK_SOLVER_DOUBLE_H
#define ILQGAMES_SOLVER_LQ_FEEDBACK_SOLVER_DOUBLE_H

#include <ilqgames/dynamics/multi_player_integrable_system_double.h>
#include <ilqgames/solver/lq_solver_double.h>
#include <ilqgames/utils/linear_dynamics_approximation_double.h>
#include <ilqgames/utils/quadratic_cost_approximation_double.h>
#include <ilqgames/utils/strategy_double.h>

#include <vector>
#include <cassert>

namespace ilqgames {

class LQFeedbackSolverDouble : public LQSolverDouble {
 public:
  ~LQFeedbackSolverDouble() {}
  LQFeedbackSolverDouble(
      const std::shared_ptr<const MultiPlayerIntegrableSystemDouble>& dynamics,
      size_t num_time_steps, bool adaptive_regularization = true)
      : LQSolverDouble(dynamics, num_time_steps),
        adaptive_regularization_(adaptive_regularization) {
    // Cache the total number of control dimensions, since this is inefficient
    // to compute.
    const Dimension total_udim = dynamics_->TotalUDim();  // 2

    // Preallocate memory for coupled Riccati solve at each time step and make
    // Eigen::Refs to the solution.
    S_.resize(total_udim, total_udim);
    X_.resize(total_udim, dynamics_->XDim() + 1);
    Y_.resize(total_udim, dynamics_->XDim() + 1);

    Dimension cumulative_udim = 0;
    for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++) {
      Ps_.push_back(
          X_.block(cumulative_udim, 0, dynamics_->UDim(ii), dynamics_->XDim()));
      alphas_.push_back(X_.col(dynamics_->XDim())
                            .segment(cumulative_udim, dynamics_->UDim(ii)));
      // Increment cumulative_udim.
      cumulative_udim += dynamics_->UDim(ii);
    }

    // Initialize Zs and zetas for each time and player. Note that we need to
    // store over all time to compute optimal costates if desired.
    Zs_.resize(num_time_steps_);
    zetas_.resize(num_time_steps_);
    for (size_t kk = 0; kk < num_time_steps_; kk++) {
      Zs_[kk].resize(dynamics_->NumPlayers());
      zetas_[kk].resize(dynamics_->NumPlayers());
      for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++) {
        Zs_[kk][ii].resize(dynamics_->XDim(), dynamics_->XDim());
        zetas_[kk][ii].resize(dynamics_->XDim());
      }
    }

    // Preallocate memory for intermediate variables F, beta.
    F_.resize(dynamics_->XDim(), dynamics_->XDim());
    beta_.resize(dynamics_->XDim());
  }

  // Solve underlying LQ game to a feedback Nash equilibrium.
  // Optionally return delta xs and costates.
  std::vector<StrategyDouble> Solve(
      const std::vector<LinearDynamicsApproximationDouble>& linearization,
      const std::vector<std::vector<QuadraticCostApproximationDouble>>&
          quadraticization,
      const VectorXd& x0, std::vector<VectorXd>* delta_xs = nullptr,
      std::vector<std::vector<VectorXd>>* costates = nullptr);

 private:
  // Quadratic/linear components of value function at the current time step in
  // the dynamic program.
  // NOTE: since these will be computed by solving a big
  // linear matrix equation S [Ps, alphas] = [YPs, Yalphas] (i.e., S X = Y), we
  // will pre-allocate the memory for that equation and define these components
  // as Eigen::Refs.
  MatrixXd S_, X_, Y_;
  std::vector<Eigen::Ref<MatrixXd>> Ps_;
  std::vector<Eigen::Ref<VectorXd>> alphas_;

  // Initialize Zs and zetas for each time and player.
  std::vector<std::vector<MatrixXd>> Zs_;
  std::vector<std::vector<VectorXd>> zetas_;

  // Preallocate memory for intermediate variables F, beta.
  MatrixXd F_;
  VectorXd beta_;

  // Adaptive regularization using Gershgorin circle theorem.
  const bool adaptive_regularization_;
};  // LQFeedbackSolverDouble

}  // namespace ilqgames

#endif
