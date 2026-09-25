#include <ilqgames/gui/control_sliders_double.h>
#include <ilqgames/utils/solver_log_double.h>
#include <ilqgames/utils/types.h>

#include <imgui/imgui.h>
#include <memory>
#include <vector>

namespace ilqgames {

void ControlSlidersDouble::Render() {
    ImGui::Begin("Control Sliders");

    // Combo box to select problem.
    if (ImGui::BeginCombo("Problem", std::to_string(prob_index_ + 1).c_str())) {
        for (size_t problem_idx = 0; problem_idx < NumProblems(); problem_idx++) {
            const bool is_selected = (prob_index_ == problem_idx);
            if (ImGui::Selectable(std::to_string(problem_idx + 1).c_str(), is_selected))
                prob_index_ = problem_idx;
            if (is_selected) ImGui::SetItemDefaultFocus();
        }

        ImGui::EndCombo();
    }

    // Make a slider to get the desired log index from a receding horizon problem.
    if (!logs_for_each_problem_.empty())
        ImGui::SliderInt("Log Index", &log_index_, 0, max_log_index_);

    // Compute endpoints of sliders to come.
    int max_solver_iterates = 0;
    Time max_final_time = -std::numeric_limits<Time>::infinity();
    Time min_initial_time = std::numeric_limits<Time>::infinity();
    for (size_t ii = 0; ii < logs_for_each_problem_.size(); ii++) {
        const auto& logs = logs_for_each_problem_[ii];

        max_solver_iterates = std::max(max_solver_iterates, static_cast<int>(logs[LogIndex(ii)]->NumIterates()) - 1);
        max_final_time = std::max(max_final_time, logs[LogIndex(ii)]->FinalTime());
        min_initial_time = std::min(min_initial_time, logs[LogIndex(ii)]->InitialTime());
    }

    // Make a slider to get the desired iterate.
    ImGui::SliderInt("Iterate", &solver_iterate_, 0, max_solver_iterates);

    // Make a slider to get the desired interpolation time.
    ImGui::SliderFloat("Interpolation Time (s)", &interpolation_time_, min_initial_time, max_final_time);

    ImGui::End();
}

}    // namespace ilqgames
