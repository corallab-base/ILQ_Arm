#ifndef ILQGAMES_UTILS_PLAYER_COST_CACHE_DOUBLE_H
#define ILQGAMES_UTILS_PLAYER_COST_CACHE_DOUBLE_H

#include <ilqgames/cost/player_cost_double.h>
#include <ilqgames/dynamics/multi_player_flat_system_double.h>
#include <ilqgames/utils/operating_point_double.h>
#include <ilqgames/utils/solver_log_double.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace ilqgames {

class PlayerCostCacheDouble {
 public:
  ~PlayerCostCacheDouble() {}
  PlayerCostCacheDouble(const std::shared_ptr<const SolverLogDouble>& log,
                  const std::vector<PlayerCostDouble>& player_costs);

  // Interpolate the given cost at the specified iterate and time.
  double Interpolate(size_t iterate, Time t, PlayerIndex player,
                    const std::string& name) const;

  // Accessors.
  const SolverLogDouble& Log() const { return *log_; }
  size_t NumPlayers() const { return evaluated_player_costs_.size(); }
  size_t NumCosts(PlayerIndex player) const {
    return evaluated_player_costs_[player].size();
  }
  bool PlayerHasCost(PlayerIndex player, const std::string& name) const {
    return evaluated_player_costs_[player].count(name) > 0;
  }
  const std::unordered_map<std::string, std::vector<std::vector<double>>>&
  EvaluatedCosts(PlayerIndex player) const {
    return evaluated_player_costs_[player];
  }
  const std::vector<double>& EvaluatedCost(size_t iterate, PlayerIndex player,
                                          const std::string& name) const {
    CHECK(PlayerHasCost(player, name));
    CHECK_LT(iterate, evaluated_player_costs_[player].at(name).size());

    return evaluated_player_costs_[player].at(name)[iterate];
  }

 private:
  // Log. Currently only used for converting between times and time steps.
  const std::shared_ptr<const SolverLogDouble> log_;

  // Player costs (indexed by player and string ID) evaluated every iterate
  // and time step.
  std::vector<std::unordered_map<std::string, std::vector<std::vector<double>>>>
      evaluated_player_costs_;
};  // class PlayerCostCacheDouble

}  // namespace ilqgames

#endif
