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

// Compute cost of a set of strategies for each player.
std::vector<double> ComputeStrategyCostsDouble(
    const std::vector<PlayerCostDouble>& player_costs,
    const std::vector<StrategyDouble>& strategies,
    const OperatingPointDouble& operating_point,
    const MultiPlayerIntegrableSystemDouble& dynamics, const VectorXd& x0,
    bool open_loop) {
  // Start at the initial state.
  VectorXd x(x0);
  Time t = 0.0;

  // Walk forward along the trajectory and accumulate total cost.
  std::vector<VectorXd> us(dynamics.NumPlayers());
  std::vector<double> total_costs(dynamics.NumPlayers(), 0.0);
  const size_t num_time_steps =
      (open_loop) ? strategies[0].Ps.size() - 1 : strategies[0].Ps.size();
  for (size_t kk = 0; kk < num_time_steps; kk++) {
    // Update controls.
    for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++) {
      if (open_loop)
        us[ii] = strategies[ii](kk, VectorXd::Zero(x.size()),
                                operating_point.us[kk][ii]);
      else
        us[ii] = strategies[ii](kk, x - operating_point.xs[kk],
                                operating_point.us[kk][ii]);
    }

    const VectorXd next_x = dynamics.Integrate(t, time::kTimeStep, x, us);
    const Time next_t = t + time::kTimeStep;

    // Update costs.
    for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++) {
      const double cost =
          (open_loop) ? player_costs[ii].EvaluateOffset(t, next_t, next_x, us)
                      : player_costs[ii].Evaluate(t, x, us);

      total_costs[ii] += cost;
    }

    // Update state and time
    x = next_x;
    t = next_t;
  }

  return total_costs;
}

std::vector<double> ComputeStrategyCostsDouble(const ProblemDouble& problem,
                                        bool open_loop) {
  return ComputeStrategyCostsDouble(
      problem.PlayerCosts(), problem.CurrentStrategies(),
      problem.CurrentOperatingPoint(), *problem.Dynamics(),
      problem.InitialState(), open_loop);
}

}  // namespace ilqgames
