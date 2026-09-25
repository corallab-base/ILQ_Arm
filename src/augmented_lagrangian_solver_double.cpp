#include <ilqgames/dynamics/multi_player_dynamical_system_double.h>
#include <ilqgames/dynamics/multi_player_integrable_system_double.h>
#include <ilqgames/solver/augmented_lagrangian_solver_double.h>
#include <ilqgames/solver/game_solver_double.h>
#include <ilqgames/solver/ilq_solver_double.h>
#include <ilqgames/solver/lq_feedback_solver_double.h>
#include <ilqgames/solver/lq_open_loop_solver_double.h>
#include <ilqgames/solver/lq_solver_double.h>
#include <ilqgames/solver/problem_double.h>
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

std::shared_ptr<SolverLogDouble> AugmentedLagrangianSolverDouble::Solve(bool* success, Time max_runtime) {
    if (success) *success = true;

    // Cache initial problem solution so we can restore it at the end.
    const auto& initial_op = problem_->CurrentOperatingPoint();
    const auto& initial_strategies = problem_->CurrentStrategies();

    // Create new log.
    std::shared_ptr<SolverLogDouble> log = CreateNewLog();

    // Determine how much time should be allocated for any individual lower level
    // solver call.
    const Time max_runtime_unconstrained_problem =
            (problem_->IsConstrained())
                ? max_runtime / static_cast<Time>(params_.max_solver_iters)
                : max_runtime;

    // Solve unconstrained problem.
    bool unconstrained_success = false;
    const auto unconstrained_log = unconstrained_solver_->Solve(
        &unconstrained_success, max_runtime_unconstrained_problem);
    log->AddLog(*unconstrained_log);

    VLOG_IF(2, !unconstrained_success)
        << "Unconstrained solver failed on first call.";
    VLOG_IF(2, unconstrained_success)
        << "Unconstrained solver succeeded on first call.";
    if (success) *success &= unconstrained_success;

    // Exit if problem is unconstrained.
    if (!problem_->IsConstrained()) return log;

    // Run until convergence or until the time runs out.
    Time elapsed = max_runtime_unconstrained_problem;
    float max_constraint_error = constants::kInfinity;
    while (log->NumIterates() < params_.max_solver_iters &&
            max_constraint_error > params_.constraint_error_tolerance &&
            elapsed < max_runtime - timer_.RuntimeUpperBound()) {
        // Start loop timer.
        timer_.Tic();

        // Increment multiplers in player costs, and in parallel compute the total
        // squared constraint error.
        max_constraint_error = -constants::kInfinity;
        const OperatingPointDouble& op = log->FinalOperatingPoint();
        for (auto& pc : problem_->PlayerCosts()) {
            for (size_t kk = 0; kk < op.xs.size(); kk++) {
                const Time t = op.t0 + time::kTimeStep * static_cast<float>(kk);
                const auto& x = op.xs[kk];
                const auto& us = op.us[kk];

                // Scale each lambda.
                for (const auto& constraint : pc.StateConstraints()) {
                    const float constraint_error = constraint->Evaluate(t, x);
                    max_constraint_error = std::max(max_constraint_error, constraint_error);
                    constraint->IncrementLambda(t, constraint_error);
                }

                for (const auto& pair : pc.ControlConstraints()) {
                    const float constraint_error =
                            pair.second->Evaluate(t, us[pair.first]);
                    max_constraint_error = std::max(max_constraint_error, constraint_error);
                    pair.second->IncrementLambda(t, constraint_error);
                }
            }
        }

        // Scale mu.
        ConstraintDouble::ScaleMu(params_.geometric_mu_scaling);

        // Log squared constraint violation.
        VLOG(2) << "Max constraint violation at iteration " << log->NumIterates()
                << " is " << max_constraint_error;

        // Update problem solution to make sure we pick up where we left off if the
        // previous unconstrained solver succeeded.
        if (unconstrained_success) {
            problem_->OverwriteSolution(log->FinalOperatingPoint(), log->FinalStrategies());
        }

        // Run unconstrained solver to convergence. Since we will update problem
        // solutions at each outer iteration, the unconstrained solver should
        // automatically start where it left off.
        const auto unconstrained_log = unconstrained_solver_->Solve(
                &unconstrained_success, max_runtime_unconstrained_problem);

        VLOG_IF(2, unconstrained_success)
                << "Unconstrained solver succeeded on iteration " << log->NumIterates();

        // If we failed then downscale all lambdas and mus for next iteration.
        if (!unconstrained_success) {
            VLOG(2) << "Unconstrained solver failed at iteration "
                            << log->NumIterates();
            VLOG(2) << "Downscaling all multipliers.";
            for (auto& pc : problem_->PlayerCosts()) {
                for (const auto& constraint : pc.StateConstraints())
                    constraint->ScaleLambdas(params_.geometric_lambda_downscaling);
                for (const auto& pair : pc.ControlConstraints())
                    pair.second->ScaleLambdas(params_.geometric_lambda_downscaling);
            }

            ConstraintDouble::ScaleMu(params_.geometric_mu_downscaling);
        }

        if (success) *success &= unconstrained_success;
        log->AddLog(*unconstrained_log);

        // Record loop time.
        elapsed += timer_.Toc();
    }

    // If we're still failing constraint satisfaction check mark as failure.
    if (max_constraint_error > params_.constraint_error_tolerance) {
        LOG(WARNING) << "Solver could not satisfy all constraints.";
        if (success) *success = false;
    }

    // Maybe restore initial solution to this problem.
    if (params_.reset_problem)
        problem_->OverwriteSolution(initial_op, initial_strategies);

    // Reset all multipliers.
    if (params_.reset_lambdas) {
        for (auto& pc : problem_->PlayerCosts()) {
            for (const auto& constraint : pc.StateConstraints())
                constraint->ScaleLambdas(constants::kDefaultLambda);
            for (const auto& pair : pc.ControlConstraints())
                pair.second->ScaleLambdas(constants::kDefaultLambda);
        }
    }

    if (params_.reset_mu) ConstraintDouble::GlobalMu() = constants::kDefaultMu;

    return log;
}

}    // namespace ilqgames
