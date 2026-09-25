#ifndef ILQGAMES_SOLVER_LQ_OPEN_LOOP_SOLVER_DOUBLE_H
#define ILQGAMES_SOLVER_LQ_OPEN_LOOP_SOLVER_DOUBLE_H

#include <ilqgames/dynamics/multi_player_integrable_system_double.h>
#include <ilqgames/solver/lq_solver_double.h>
#include <ilqgames/utils/linear_dynamics_approximation_double.h>
#include <ilqgames/utils/quadratic_cost_approximation_double.h>
#include <ilqgames/utils/strategy_double.h>
#include <vector>

namespace ilqgames {

class LQOpenLoopSolverDouble : public LQSolverDouble {
 public:
  ~LQOpenLoopSolverDouble() {}
  LQOpenLoopSolverDouble(
      const std::shared_ptr<const MultiPlayerIntegrableSystemDouble>& dynamics,
      size_t num_time_steps)
      : LQSolverDouble(dynamics, num_time_steps) {
    // Initialize Ms and ms.
    Ms_.resize(num_time_steps_);
    ms_.resize(num_time_steps_);
    for (size_t kk = 0; kk < num_time_steps_; kk++) {
      Ms_[kk].resize(dynamics_->NumPlayers(),
                     MatrixXd::Zero(dynamics_->XDim(), dynamics_->XDim()));
      ms_[kk].resize(dynamics_->NumPlayers(),
                     VectorXd::Zero(dynamics_->XDim()));
    }

    // Initialize other "special" terms and decompositions.
    intermediate_terms_.resize(num_time_steps_ - 1,
                               VectorXd::Zero(dynamics_->XDim()));
    capital_lambdas_.resize(
        num_time_steps_ - 1,
        MatrixXd::Zero(dynamics_->XDim(), dynamics_->XDim()));
    qr_capital_lambdas_.resize(
        num_time_steps_ - 1,
        Eigen::HouseholderQR<MatrixXd>(dynamics_->XDim(), dynamics_->XDim()));

    std::vector<Eigen::LDLT<MatrixXd>> chol_Rs_element;
    std::vector<MatrixXd> warped_Bs_element;
    std::vector<VectorXd> warped_rs_element;
    for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++) {
      chol_Rs_element.emplace_back(dynamics_->UDim(ii));
      warped_Bs_element.emplace_back(dynamics_->UDim(ii), dynamics_->XDim());
      warped_rs_element.emplace_back(dynamics_->UDim(ii));
    }

    chol_Rs_.resize(num_time_steps_ - 1, chol_Rs_element);
    warped_Bs_.resize(num_time_steps_ - 1, warped_Bs_element);
    warped_rs_.resize(num_time_steps_ - 1, warped_rs_element);
  }

  // Solve underlying LQ game to a open-loop Nash equilibrium.
  // Optionally return delta xs and costates.
  std::vector<StrategyDouble> Solve(
      const std::vector<LinearDynamicsApproximationDouble>& linearization,
      const std::vector<std::vector<QuadraticCostApproximationDouble>>&
          quadraticization,
      const VectorXd& x0, std::vector<VectorXd>* delta_xs = nullptr,
      std::vector<std::vector<VectorXd>>* costates = nullptr);

 private:
  // Initialize Ms and ms.
  std::vector<std::vector<VectorXd>> ms_;
  std::vector<std::vector<MatrixXd>> Ms_;

  // Instantiate the rest of the "special" terms and decompositions.
  std::vector<VectorXd> intermediate_terms_;
  std::vector<MatrixXd> capital_lambdas_;
  std::vector<Eigen::HouseholderQR<MatrixXd>> qr_capital_lambdas_;
  std::vector<std::vector<Eigen::LDLT<MatrixXd>>> chol_Rs_;
  std::vector<std::vector<MatrixXd>> warped_Bs_;
  std::vector<std::vector<VectorXd>> warped_rs_;

};  // LQOpenLoopSolverDouble

}  // namespace ilqgames

#endif
