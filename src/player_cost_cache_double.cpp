#include <ilqgames/cost/player_cost_double.h>
#include <ilqgames/dynamics/multi_player_flat_system_double.h>
#include <ilqgames/utils/operating_point_double.h>
#include <ilqgames/utils/player_cost_cache_double.h>
#include <ilqgames/utils/solver_log_double.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace ilqgames {

PlayerCostCacheDouble::PlayerCostCacheDouble(const std::shared_ptr<const SolverLogDouble>& log,
                                            const std::vector<PlayerCostDouble>& player_costs)
        : log_(log) {
    CHECK_NOTNULL(log.get());

    // Populate costs separately for each player.
    evaluated_player_costs_.resize(player_costs.size());
    for (PlayerIndex ii = 0; ii < player_costs.size(); ii++) {
        const auto& player_cost = player_costs[ii];
        auto& evaluated_costs = evaluated_player_costs_[ii];

        // Cycle through each separate cost.
        // Start with state costs.
        for (const auto& cost : player_cost.StateCosts()) {
            auto e = evaluated_costs.emplace(cost->Name(), std::vector<std::vector<double>>());
            LOG_IF(WARNING, !e.second)
                    << "Player " << ii
                    << " has duplicate cost with name: " << cost->Name();

            auto& entry = e.first->second;
            entry.resize(log->NumIterates());
            for (size_t jj = 0; jj < log->NumIterates(); jj++) {
                entry[jj].resize(time::kNumTimeSteps);

                for (size_t kk = 0; kk < time::kNumTimeSteps; kk++) {
                    const VectorXd x = log->State(jj, kk);
                    entry[jj][kk] = cost->Evaluate(log->IndexToTime(kk), x);
                }
            }
        }

        // Now handle control costs.
        for (const auto& cost_pair : player_cost.ControlCosts()) {
            const auto other_player = cost_pair.first;
            const auto& cost = cost_pair.second;
            auto e = evaluated_costs.emplace(cost->Name(), std::vector<std::vector<double>>());
            LOG_IF(WARNING, !e.second)
                    << "Player " << ii
                    << " has duplicate cost with name: " << cost->Name();

            auto& entry = e.first->second;
            entry.resize(log->NumIterates());
            for (size_t jj = 0; jj < log->NumIterates(); jj++) {
                entry[jj].resize(time::kNumTimeSteps);

                for (size_t kk = 0; kk < time::kNumTimeSteps; kk++) {
                    entry[jj][kk] = cost->Evaluate(log->IndexToTime(kk), log->Control(jj, kk, other_player));
                }
            }
        }

        // Handle constraints.
        for (const auto& constraint : player_cost.StateConstraints()) {
            auto e = evaluated_costs.emplace(constraint->Name(), std::vector<std::vector<double>>());
            LOG_IF(WARNING, !e.second)
                    << "Player " << ii
                    << " has duplicate constraint with name: " << constraint->Name();

            auto& entry = e.first->second;
            entry.resize(log->NumIterates());
            for (size_t jj = 0; jj < log->NumIterates(); jj++) {
                entry[jj].resize(time::kNumTimeSteps);

                for (size_t kk = 0; kk < time::kNumTimeSteps; kk++) {
                    const VectorXd x = log->State(jj, kk);
                    entry[jj][kk] = constraint->Evaluate(log->IndexToTime(kk), x);
                }
            }
        }

        // Now handle control constraints.
        for (const auto& constraint_pair : player_cost.ControlConstraints()) {
            const auto other_player = constraint_pair.first;
            const auto& constraint = constraint_pair.second;
            auto e = evaluated_costs.emplace(constraint->Name(), std::vector<std::vector<double>>());
            LOG_IF(WARNING, !e.second)
                    << "Player " << ii
                    << " has duplicate constraint with name: " << constraint->Name();

            auto& entry = e.first->second;
            entry.resize(log->NumIterates());
            for (size_t jj = 0; jj < log->NumIterates(); jj++) {
                entry[jj].resize(time::kNumTimeSteps);

                for (size_t kk = 0; kk < time::kNumTimeSteps; kk++) {
                    entry[jj][kk] = constraint->Evaluate(
                            log->IndexToTime(kk), log->Control(jj, kk, other_player));
                }
            }
        }
    }
}

double PlayerCostCacheDouble::Interpolate(size_t iterate, Time t, PlayerIndex player,
                                            const std::string& name) const {
    CHECK_LT(iterate, log_->NumIterates());
    CHECK_LT(player, evaluated_player_costs_.size());

    // Access the approprate time-indexed list of costs.
    const auto& costs = evaluated_player_costs_[player].at(name)[iterate];

    // Interpolate this list.
    const size_t lo = log_->TimeToIndex(t);
    const size_t hi = std::min(lo + 1, time::kNumTimeSteps - 1);

    const double frac = (t - log_->IndexToTime(lo)) / time::kTimeStep;
    return (1.0 - frac) * costs[lo] + frac * costs[hi];
}

}    // namespace ilqgames
