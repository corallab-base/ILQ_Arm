#include <ilqgames/solver/lq_feedback_solver_double.h>
#include <ilqgames/utils/linear_dynamics_approximation_double.h>
#include <ilqgames/utils/quadratic_cost_approximation_double.h>
#include <ilqgames/utils/strategy_double.h>

#include <glog/logging.h>
#include <vector>

namespace ilqgames {

std::vector<StrategyDouble> LQFeedbackSolverDouble::Solve(
        const std::vector<LinearDynamicsApproximationDouble>& linearization,
        const std::vector<std::vector<QuadraticCostApproximationDouble>>& quadraticization,
        const VectorXd& x0, std::vector<VectorXd>* delta_xs,
        std::vector<std::vector<VectorXd>>* costates) {
    CHECK_EQ(linearization.size(), num_time_steps_);
    CHECK_EQ(quadraticization.size(), num_time_steps_);

    // Make sure delta_xs and costates are the right size.
    if (delta_xs) CHECK_NOTNULL(costates);
    if (costates) CHECK_NOTNULL(delta_xs);
    if (delta_xs) {
        delta_xs->resize(num_time_steps_);
        costates->resize(num_time_steps_);
        for (size_t kk = 0; kk < num_time_steps_; kk++) {
            (*delta_xs)[kk].resize(dynamics_->XDim());
            (*costates)[kk].resize(dynamics_->NumPlayers());
            for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++)
                (*costates)[kk][ii].resize(dynamics_->XDim());
        }
    }

    // List of player-indexed strategies (each of which is a time-indexed
    // affine state error-feedback controller).
    std::vector<StrategyDouble> strategies;
    for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++)
        strategies.emplace_back(num_time_steps_, dynamics_->XDim(), dynamics_->UDim(ii));

    // Initialize Zs and zetas at the final time.
    for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++) {
        Zs_[num_time_steps_ - 1][ii] = quadraticization.back()[ii].state.hess;
        zetas_[num_time_steps_ - 1][ii] = quadraticization.back()[ii].state.grad;
    }

    // Work backward in time and solve the dynamic program.
    // NOTE: time starts from the second-to-last entry since we'll treat the final
    // entry as a terminal cost as in Basar and Olsder, ch. 6.
    for (int kk = num_time_steps_ - 2; kk >= 0; kk--) {
        // Unpack linearization and quadraticization at this time step.
        const auto& lin = linearization[kk];
        const auto& quad = quadraticization[kk];

        // Populate coupling matrix S for linear matrix equation to determine X (Ps
        // and alphas).
        // NOTE: S is generally dense and asymmetric, though it is symmetric if all
        // players have the same Z.
        Dimension cumulative_udim_row = 0;
        for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++) {
            // Intermediate variable to store B[ii]' * Z[ii].
            const MatrixXd BiZi = lin.Bs[ii].transpose() * Zs_[kk + 1][ii];

            Dimension cumulative_udim_col = 0;
            for (PlayerIndex jj = 0; jj < dynamics_->NumPlayers(); jj++) {
                Eigen::Ref<MatrixXd> S_block =
                        S_.block(cumulative_udim_row, cumulative_udim_col,
                                         dynamics_->UDim(ii), dynamics_->UDim(jj));

                if (ii == jj) {
                    // Does player ii's cost depend upon player jj's control?
                    const auto control_iter = quad[ii].control.find(ii);
                    CHECK(control_iter != quad[ii].control.end())
                            << "Player " << ii << " is missing a control Hessian.";

                    S_block = BiZi * lin.Bs[ii] + control_iter->second.hess;
                } else {
                    S_block = BiZi * lin.Bs[jj];
                }

                // Increment cumulative_udim_col.
                cumulative_udim_col += dynamics_->UDim(jj);
            }

            // Set appropriate blocks of Y.
            Y_.block(cumulative_udim_row, 0, dynamics_->UDim(ii), dynamics_->XDim()) =
                    BiZi * lin.A;
            Y_.col(dynamics_->XDim())
                    .segment(cumulative_udim_row, dynamics_->UDim(ii)) =
                    lin.Bs[ii].transpose() * zetas_[kk + 1][ii] +
                    quad[ii].control.at(ii).grad;

            // Increment cumulative_udim_row.
            cumulative_udim_row += dynamics_->UDim(ii);
        }

        if (adaptive_regularization_) {
            // Regularize `S` to have positive eigenvalues using the Gershgorin circle
            // theorem (https://en.wikipedia.org/wiki/Gershgorin_circle_theorem). That
            // is, for column i, compute the 1-norm of non-diagonal entries and ensure
            // that the ii^th entry of `S` is greater than that norm by adding some
            // amount to that diagonal entry.
            for (size_t ii = 0; ii < S_.cols(); ii++) {
                const float radius = S_.col(ii).lpNorm<1>() - std::abs(S_(ii, ii));
                const float eval_lo = S_(ii, ii) - radius;

                constexpr float min_eval = 1e-3;
                if (eval_lo < min_eval) S_(ii, ii) += radius + min_eval;
            }
        }

        // Solve linear matrix equality S X = Y.
        // NOTE: not 100% sure that this avoids dynamic memory allocation.
        X_ = S_.householderQr().solve(Y_);

        // Set strategy at current time step.
        for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++) {
            strategies[ii].Ps[kk] = Ps_[ii];
            strategies[ii].alphas[kk] = alphas_[ii];
        }

        // Compute F and beta.
        F_ = lin.A;
        beta_ = VectorXd::Zero(dynamics_->XDim());
        for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++) {
            F_ -= lin.Bs[ii] * Ps_[ii];
            beta_ -= lin.Bs[ii] * alphas_[ii];
        }

        // Update Zs and zetas.
        for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++) {
            zetas_[kk][ii] =
                    (F_.transpose() * (zetas_[kk + 1][ii] + Zs_[kk + 1][ii] * beta_) +
                     quad[ii].state.grad).eval();
            Zs_[kk][ii] =
                    (F_.transpose() * Zs_[kk + 1][ii] * F_ + quad[ii].state.hess).eval();

            // Add terms for nonzero Rijs.
            for (const auto& Rij_entry : quad[ii].control) {
                const PlayerIndex jj = Rij_entry.first;
                const MatrixXd& Rij = Rij_entry.second.hess;
                const VectorXd& rij = Rij_entry.second.grad;
                zetas_[kk][ii] += Ps_[jj].transpose() * (Rij * alphas_[jj] - rij);
                Zs_[kk][ii] += Ps_[jj].transpose() * Rij * Ps_[jj];
            }
        }
    }

    // Maybe compute delta_xs and costates forward in time.
    if (delta_xs) {
        VectorXd x_star = x0;
        VectorXd last_x_star;
        for (size_t kk = 0; kk < num_time_steps_; kk++) {
            (*delta_xs)[kk] = x_star;
            for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++) {
                if (kk < num_time_steps_ - 1)
                    (*costates)[kk][ii] = -Zs_[kk + 1][ii] * x_star - zetas_[kk + 1][ii];
                else
                    (*costates)[kk][ii].setZero();
            }

            // Unpack linearization at this time step.
            const auto& lin = linearization[kk];

            // Compute optimal x.
            last_x_star = x_star;
            x_star = lin.A * last_x_star;
            for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++)
                x_star -= lin.Bs[ii] * strategies[ii].alphas[kk];
        }
    }

    return strategies;
}

}    // namespace ilqgames
