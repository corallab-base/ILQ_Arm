#ifndef ILQGAMES_GUI_CONTROL_SLIDERS_DOUBLE_H
#define ILQGAMES_GUI_CONTROL_SLIDERS_DOUBLE_H

#include <ilqgames/utils/solver_log_double.h>

#include <memory>
#include <vector>

namespace ilqgames {

class ControlSlidersDouble {
 public:
  ~ControlSlidersDouble() {}
  ControlSlidersDouble(const std::vector<
                 std::vector<std::shared_ptr<const ilqgames::SolverLogDouble>>>&
                     logs_for_each_problem)
      : interpolation_time_(0.0),
        solver_iterate_(0),
        log_index_(0),
        max_log_index_(0),
        prob_index_(0),
        logs_for_each_problem_(logs_for_each_problem) {
    for (const auto& logs : logs_for_each_problem_) {
      for (const auto& log : logs) CHECK_NOTNULL(log.get());
    }

    // Compute max log index.
    for (const auto& logs : logs_for_each_problem_) {
      if (logs.size() > static_cast<size_t>(max_log_index_) + 1)
        max_log_index_ = logs.size() - 1;
    }
  }

  // Render all the sliders in a separate window.
  void Render();

  // Accessors.
  size_t NumProblems() const { return logs_for_each_problem_.size(); }
  const std::vector<std::vector<std::shared_ptr<const SolverLogDouble>>>&
  LogsForEachProblem() const {
    return logs_for_each_problem_;
  }
  std::vector<std::shared_ptr<const SolverLogDouble>> LogForEachProblem() const {
    std::vector<std::shared_ptr<const SolverLogDouble>> logs(NumProblems());
    for (size_t ii = 0; ii < NumProblems(); ii++)
      logs[ii] = logs_for_each_problem_[ii][LogIndex(ii)];
    return logs;
  }

  Time InterpolationTime(size_t problem_idx) const {
    CHECK_LT(problem_idx, NumProblems());

    const auto& logs = logs_for_each_problem_[problem_idx];
    const int log_idx = LogIndex(problem_idx);
    return std::max(std::min(static_cast<Time>(interpolation_time_),
                             logs[log_idx]->FinalTime()),
                    logs[log_idx]->InitialTime());
  }
  int SolverIterate(size_t problem_idx) const {
    CHECK_LT(problem_idx, NumProblems());

    const auto& logs = logs_for_each_problem_[problem_idx];
    const int log_idx = LogIndex(problem_idx);
    return std::min(solver_iterate_,
                    static_cast<int>(logs[log_idx]->NumIterates() - 1));
  }
  int LogIndex(size_t problem_idx) const {
    CHECK_LT(problem_idx, NumProblems());

    const auto& logs = logs_for_each_problem_[problem_idx];
    return std::min(log_index_, static_cast<int>(logs.size() - 1));
  }
  int MaxLogIndex() const { return max_log_index_; }

  size_t ProbIndex() const { return prob_index_; }

 private:
  // Time at which to interpolate each trajectory.
  float interpolation_time_;

  // Solver iterate to display.
  int solver_iterate_;

  // Log index to render for receding horizon problems.
  int log_index_;

  size_t prob_index_;

  // Keep track of the max number of log indices across all problems.
  int max_log_index_;

  // List of all logs we might want to inspect, indexed by problem, then by
  // receding horizon invocation.
  const std::vector<std::vector<std::shared_ptr<const ilqgames::SolverLogDouble>>>
      logs_for_each_problem_;
};  // class ControlSlidersDouble

}  // namespace ilqgames

#endif
