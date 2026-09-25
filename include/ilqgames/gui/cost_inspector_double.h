#ifndef ILQGAMES_GUI_COST_INSPECTOR_DOUBLE_H
#define ILQGAMES_GUI_COST_INSPECTOR_DOUBLE_H

#include <ilqgames/cost/player_cost_double.h>
#include <ilqgames/dynamics/multi_player_flat_system_double.h>
#include <ilqgames/gui/control_sliders_double.h>
#include <ilqgames/utils/operating_point_double.h>
#include <ilqgames/utils/player_cost_cache_double.h>
#include <ilqgames/utils/solver_log_double.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <imgui/imgui.h>
#include <string>
#include <vector>

namespace ilqgames {

class CostInspectorDouble {
 public:
  ~CostInspectorDouble() {}

  // Takes in a log and lists of x/y/heading indices in
  // the state vector.
  CostInspectorDouble(const std::shared_ptr<const ControlSlidersDouble>& sliders,
                const std::vector<std::vector<PlayerCostDouble>>& player_costs)
      : sliders_(sliders),
        selected_problem_(0),
        selected_player_(0),
        selected_cost_name_("<Please select a cost>") {
    CHECK_NOTNULL(sliders_.get());
    CHECK_EQ(player_costs.size(), sliders_->NumProblems());

    player_costs_.resize(sliders_->NumProblems());
    for (size_t problem_idx = 0; problem_idx < sliders_->NumProblems();
         problem_idx++) {
      for (const auto& log : sliders_->LogsForEachProblem()[problem_idx])
        player_costs_[problem_idx].emplace_back(log, player_costs[problem_idx]);
    }
  }

  // Render the appropriate costs.
  void Render() const;

 private:
  // Control sliders.
  const std::shared_ptr<const ControlSlidersDouble> sliders_;

  // Player cost cache for each log, for each problem.
  std::vector<std::vector<PlayerCostCacheDouble>> player_costs_;

  // Currently selected problem, player and cost name.
  mutable size_t selected_problem_;
  mutable PlayerIndex selected_player_;
  mutable std::string selected_cost_name_;
};  // class CostInspectorDouble

}  // namespace ilqgames

#endif
