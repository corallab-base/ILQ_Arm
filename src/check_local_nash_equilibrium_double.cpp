#include <ilqgames/cost/player_cost_double.h>
#include <ilqgames/dynamics/multi_player_flat_system_double.h>
#include <ilqgames/dynamics/multi_player_integrable_system_double.h>
#include <ilqgames/utils/compute_strategy_costs_double.h>
#include <ilqgames/utils/operating_point_double.h>
#include <ilqgames/utils/quadratic_cost_approximation_double.h>
#include <ilqgames/utils/strategy_double.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <Eigen/Dense>
#include <random>
#include <vector>

namespace ilqgames {

bool NumericalCheckLocalNashEquilibriumDouble(
    const std::vector<PlayerCostDouble>& player_costs,
    const std::vector<StrategyDouble>& strategies,
    const OperatingPointDouble& operating_point,
    const MultiPlayerIntegrableSystemDouble& dynamics, const VectorXd& x0,
    double max_perturbation, bool open_loop) {
  CHECK_EQ(strategies.size(), player_costs.size());
  CHECK_EQ(strategies.size(), dynamics.NumPlayers());
  CHECK_EQ(x0.size(), dynamics.XDim());

  const size_t num_time_steps = strategies[0].Ps.size();
  CHECK_EQ(num_time_steps, strategies[0].alphas.size());

  // Compute nominal equilibrium cost and be sure to use only 1-step Euler
  // integration.
  const bool was_integrating_using_euler =
      MultiPlayerIntegrableSystemDouble::IntegrationUsesEuler();
  if (!was_integrating_using_euler)
    MultiPlayerIntegrableSystemDouble::IntegrateUsingEuler();
  const std::vector<double> nominal_costs = ComputeStrategyCostsDouble(
      player_costs, strategies, operating_point, dynamics, x0, open_loop);

  // For each player, perturb strategies with Gaussian noise a bunch of times
  // and if cost decreases then return false.
  std::vector<StrategyDouble> perturbed_strategies_lower(strategies);
  std::vector<StrategyDouble> perturbed_strategies_upper(strategies);
  for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++) {
    for (size_t kk = 0; kk < num_time_steps - 1; kk++) {
      VectorXd& alphak_lower = perturbed_strategies_lower[ii].alphas[kk];
      VectorXd& alphak_upper = perturbed_strategies_upper[ii].alphas[kk];

      for (size_t jj = 0; jj < alphak_lower.size(); jj++) {
        alphak_lower(jj) -= max_perturbation;
        alphak_upper(jj) += max_perturbation;

        // Compute new costs.
        const std::vector<double> perturbed_costs_lower =
            ComputeStrategyCostsDouble(player_costs, perturbed_strategies_lower,
                                 operating_point, dynamics, x0, open_loop);
        const std::vector<double> perturbed_costs_upper =
            ComputeStrategyCostsDouble(player_costs, perturbed_strategies_upper,
                                 operating_point, dynamics, x0, open_loop);

        // Check Nash condition.
        if (std::min(perturbed_costs_lower[ii], perturbed_costs_upper[ii]) <
            nominal_costs[ii]) {
          // Other users will likely want RK4 integration.
          if (!was_integrating_using_euler)
            MultiPlayerIntegrableSystemDouble::IntegrateUsingRK4();
          return false;
        }

        // Reset this alpha.
        alphak_lower = strategies[ii].alphas[kk];
        alphak_upper = strategies[ii].alphas[kk];
      }
    }
  }

  // Other users will likely want RK4 integration.
  MultiPlayerIntegrableSystemDouble::IntegrateUsingRK4();
  return true;
}

bool NumericalCheckLocalNashEquilibriumDouble(const ProblemDouble& problem,
                                        double max_perturbation,
                                        bool open_loop) {
  return NumericalCheckLocalNashEquilibriumDouble(
      problem.PlayerCosts(), problem.CurrentStrategies(),
      problem.CurrentOperatingPoint(), *problem.Dynamics(),
      problem.InitialState(), max_perturbation, open_loop);
}

bool CheckSufficientLocalNashEquilibrium(
    const std::vector<PlayerCostDouble>& player_costs,
    const OperatingPointDouble& operating_point,
    const std::shared_ptr<const MultiPlayerIntegrableSystemDouble> dynamics) {
  // Unpack number of players and number of time steps.
  const PlayerIndex num_players = player_costs.size();
  const size_t num_time_steps = operating_point.xs.size();
  const Dimension xdim = operating_point.xs[0].size();

  // Set up quadratic cost approximations.
  std::vector<QuadraticCostApproximationDouble> quadraticization(
      num_players, QuadraticCostApproximationDouble(xdim));

  // Quadraticize costs and check PSD conditions.
  for (size_t kk = 0; kk < num_time_steps; kk++) {
    const Time t = operating_point.t0 + static_cast<Time>(kk) * time::kTimeStep;
    VectorXd x = operating_point.xs[kk];
    std::vector<VectorXd> us = operating_point.us[kk];

    // Maybe convert out of linear system coordinates.
    if (dynamics.get() && dynamics->TreatAsLinear()) {
      const auto& dyn =
          *static_cast<const MultiPlayerFlatSystemDouble*>(dynamics.get());

      // Previous x, us are actually xi, vs.
      x = dyn.FromLinearSystemState(x.eval());
      us = dyn.LinearizingControls(x, std::vector<VectorXd>(us));
    }

    std::transform(player_costs.begin(), player_costs.end(),
                   quadraticization.begin(),
                   [&t, &x, &us](const PlayerCostDouble& cost) {
                     return cost.Quadraticize(t, x, us);
                   });

    // Check if Q, Rs PSD.
    constexpr double kErrorMargin = 1e-4;
    for (const auto& q : quadraticization) {
      const auto eig_Q = Eigen::SelfAdjointEigenSolver<MatrixXd>(q.state.hess);
      if (eig_Q.eigenvalues().minCoeff() < -kErrorMargin) {
        return false;
      }

      for (const auto& entry : q.control) {
        const auto eig_R =
            Eigen::SelfAdjointEigenSolver<MatrixXd>(entry.second.hess);
        if (eig_R.eigenvalues().minCoeff() < -kErrorMargin) return false;
      }
    }
  }

  return true;
}

bool CheckSufficientLocalNashEquilibrium(const ProblemDouble& problem) {
  return CheckSufficientLocalNashEquilibrium(problem.PlayerCosts(),
                                             problem.CurrentOperatingPoint(),
                                             problem.Dynamics());
}

}  // namespace ilqgames
