#include <ilqgames/gui/control_sliders_double.h>
#include <ilqgames/gui/cost_inspector_double.h>
#include <ilqgames/utils/operating_point_double.h>
#include <ilqgames/utils/player_cost_cache_double.h>
#include <ilqgames/utils/solver_log_double.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <imgui/imgui.h>
#include <string>
#include <vector>

float DoubleGetter(void* data, int idx) {
      const auto* v = static_cast<const std::vector<double>*>(data);
      return static_cast<float>((*v)[idx]);
}

namespace ilqgames {

void CostInspectorDouble::Render() const {
    const auto& costs1 =
            player_costs_[selected_problem_][sliders_->LogIndex(selected_problem_)];

    if (costs1.Log().NumIterates() == 0) return;

    // Set up main window.
    ImGui::Begin("Cost Inspector");
    selected_problem_ = sliders_->ProbIndex();

    // Combo box to select player.
    const auto& costs2 =
            player_costs_[selected_problem_][sliders_->LogIndex(selected_problem_)];

    if (ImGui::BeginCombo("Player",
                                                std::to_string(selected_player_ + 1).c_str())) {
        for (PlayerIndex ii = 0; ii < costs2.NumPlayers(); ii++) {
            const bool is_selected = (selected_player_ == ii);
            if (ImGui::Selectable(std::to_string(ii + 1).c_str(), is_selected))
                selected_player_ = ii;
            if (is_selected) ImGui::SetItemDefaultFocus();
        }

        ImGui::EndCombo();
    }

    // Combo box to select cost.
    const auto& costs3 =
            player_costs_[selected_problem_][sliders_->LogIndex(selected_problem_)];

    if (ImGui::BeginCombo("Cost", selected_cost_name_.c_str())) {
        for (const auto& entry : costs3.EvaluatedCosts(selected_player_)) {
            const std::string& cost_name = entry.first;
            const bool is_selected = (selected_cost_name_ == cost_name);
            if (ImGui::Selectable(cost_name.c_str(), is_selected))
                selected_cost_name_ = cost_name;
            if (is_selected) ImGui::SetItemDefaultFocus();
        }

        ImGui::EndCombo();
    }

    // Plot the given cost.
    const auto& costs4 =
            player_costs_[selected_problem_][sliders_->LogIndex(selected_problem_)];
    if (ImGui::BeginChild("Cost over time", ImVec2(0, 0), false)) {
        const std::string label = "Player " + std::to_string(selected_player_ + 1) +
                                                            ": " + selected_cost_name_;
        if (costs4.PlayerHasCost(selected_player_, selected_cost_name_)) {
            const std::vector<double>& values =
                    costs4.EvaluatedCost(sliders_->SolverIterate(selected_problem_),
                                                             selected_player_, selected_cost_name_);

            auto value_getter = [](void* data, int idx) -> float {
                const auto& v = *static_cast<const std::vector<double>*>(data);
                return static_cast<float>(v[idx]);
            };

            float max_val = -FLT_MAX;
            float min_val = FLT_MAX;
            for (double v : values) {
                    if (v > max_val) max_val = (float)v;
                    if (v < min_val) min_val = (float)v;
            }
            // Add a tiny buffer so lines don't clip the exact edge
            if (min_val == max_val) { max_val += 5.0f; min_val -= 5.0f; }

            ImGui::PlotLines(
                    label.c_str(),
                    value_getter,
                    (void*)&values,
                    (int)values.size(),
                    0,
                    NULL,
                    min_val,
                    max_val,
                    ImGui::GetContentRegionAvail()
            );

            ImVec2 p0 = ImGui::GetItemRectMin();
            ImVec2 p1 = ImGui::GetItemRectMax();
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            float range = max_val - min_val;

            // Color for Grid/Text
            ImU32 grid_color = IM_COL32(200, 200, 200, 50); // Transparent grey
            ImU32 text_color = IM_COL32(255, 255, 255, 200);

            // Draw Zero Line
            if (min_val < 0.0f && max_val > 0.0f) {
                    float y_zero_rel = (0.0f - min_val) / range; // 0 to 1
                    float y_zero_screen = p1.y - y_zero_rel * (p1.y - p0.y);
                    
                    // Draw distinct red line for Zero
                    draw_list->AddLine(ImVec2(p0.x, y_zero_screen), ImVec2(p1.x, y_zero_screen), IM_COL32(255, 100, 100, 150));
                    draw_list->AddText(ImVec2(p0.x + 2, y_zero_screen - 15), text_color, "0.0");
            }

            // Draw Max Label
            char buf[32];
            snprintf(buf, 32, "Max: %.2f", max_val);
            draw_list->AddText(ImVec2(p0.x + 2, p0.y), text_color, buf);

            // Draw Min Label
            snprintf(buf, 32, "Min: %.2f", min_val);
            draw_list->AddText(ImVec2(p0.x + 2, p1.y - 15), text_color, buf);

            // Show a vertical line at the current time.
            const float time = sliders_->InterpolationTime(selected_problem_);
            const ImU32 color =
                    ImColor(ImVec4(234.0 / 255.0, 110.0 / 255.0, 110.0 / 255.0, 0.5));
            constexpr float kLineThickness = 2.0;

            const ImVec2 window_top_left = ImGui::GetWindowPos();
            const float line_y_lower = window_top_left.y + ImGui::GetWindowHeight();
            const float line_y_upper = window_top_left.y;
            const float line_x = window_top_left.x + ImGui::GetWindowWidth() * time / costs4.Log().FinalTime();

            draw_list->AddLine(ImVec2(line_x, line_y_lower), ImVec2(line_x, line_y_upper), color, kLineThickness);
        }
    }

    ImGui::EndChild();
    ImGui::End();
}

}    // namespace ilqgames
