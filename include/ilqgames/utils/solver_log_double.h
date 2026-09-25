#ifndef ILQGAMES_UTILS_LOG_DOUBLE_H
#define ILQGAMES_UTILS_LOG_DOUBLE_H

#include <ilqgames/utils/operating_point_double.h>
#include <ilqgames/utils/strategy_double.h>
#include <ilqgames/utils/types.h>
#include <ilqgames/utils/uncopyable.h>

#include <math.h>
#include <vector>

namespace ilqgames {

// Default experiment name to use.
std::string DefaultExperimentName();

class SolverLogDouble : private Uncopyable {
 public:
    ~SolverLogDouble() {}
    SolverLogDouble() {}

    // Add a new solver iterate.
    void AddSolverIterate(const OperatingPointDouble& operating_point,
                        const std::vector<StrategyDouble>& strategies,
                        const std::vector<double>& total_costs,
                        Time cumulative_runtime, bool was_converged) {
        operating_points_.push_back(operating_point);
        strategies_.push_back(strategies);
        total_player_costs_.push_back(total_costs);
        cumulative_runtimes_.push_back(cumulative_runtime);
        was_converged_.push_back(was_converged);
    }

    // Add a whole other log.
    void AddLog(const SolverLogDouble& log) {
        for (size_t ii = 0; ii < log.NumIterates(); ii++) {
            AddSolverIterate(log.operating_points_[ii], log.strategies_[ii],
                            log.total_player_costs_[ii],
                            log.cumulative_runtimes_[ii], log.was_converged_[ii]);
        }
    }

    // Clear all but first entry. Used by the solver to return initial conditions
    // upon failure.
    void ClearAllButFirstIterate() {
        constexpr size_t kOneIterate = 1;

        CHECK_GE(operating_points_.size(), kOneIterate);
        operating_points_.resize(kOneIterate, operating_points_.front());
        strategies_.resize(kOneIterate);
        total_player_costs_.resize(kOneIterate);
        cumulative_runtimes_.resize(kOneIterate);
        was_converged_.resize(kOneIterate);
    }

    // Accessors.
    bool WasConverged() const { return was_converged_.back(); }
    bool WasConverged(size_t idx) const { return was_converged_[idx]; }
    Time InitialTime() const {
        return (NumIterates() > 0) ? operating_points_[0].t0 : 0.0;
    }
    Time FinalTime() const {
        return (NumIterates() > 0) ? IndexToTime(operating_points_[0].xs.size() - 1)
                                                             : 0.0;
    }
    PlayerIndex NumPlayers() const { return strategies_[0].size(); }
    size_t NumIterates() const { return operating_points_.size(); }
    std::vector<double> TotalCosts() const { return total_player_costs_.back(); }

    const std::vector<StrategyDouble>& InitialStrategies() const {
        return strategies_.front();
    }
    const OperatingPointDouble& InitialOperatingPoint() const {
        return operating_points_.front();
    }
    const std::vector<StrategyDouble>& FinalStrategies() const {
        return strategies_.back();
    }
    const OperatingPointDouble& FinalOperatingPoint() const {
        return operating_points_.back();
    }

    VectorXd InterpolateState(size_t iterate, Time t) const;
    double InterpolateState(size_t iterate, Time t, Dimension dim) const;
    VectorXd InterpolateControl(size_t iterate, Time t, PlayerIndex player) const;
    double InterpolateControl(size_t iterate, Time t, PlayerIndex player,
                                                     Dimension dim) const;

    std::vector<MatrixXd> Ps(size_t iterate, size_t time_index) const;
    std::vector<VectorXd> alphas(size_t iterate, size_t time_index) const;
    MatrixXd P(size_t iterate, size_t time_index, PlayerIndex player) const;
    VectorXd alpha(size_t iterate, size_t time_index, PlayerIndex player) const;

    VectorXd State(size_t iterate, size_t time_index) const {
        return operating_points_[iterate].xs[time_index];
    }
    double State(size_t iterate, size_t time_index, Dimension dim) const {
        return operating_points_[iterate].xs[time_index](dim);
    }
    VectorXd Control(size_t iterate, size_t time_index,
                                     PlayerIndex player) const {
        return operating_points_[iterate].us[time_index][player];
    }
    double Control(size_t iterate, size_t time_index, PlayerIndex player, Dimension dim) const {
        return operating_points_[iterate].us[time_index][player](dim);
    }

    std::vector<MatrixXd> Ps(size_t iterate, Time t) const {
        return Ps(iterate, TimeToIndex(t));
    }
    std::vector<VectorXd> alphas(size_t iterate, Time t) const {
        return alphas(iterate, TimeToIndex(t));
    }
    MatrixXd P(size_t iterate, Time t, PlayerIndex player) const {
        return P(iterate, TimeToIndex(t), player);
    }
    VectorXd alpha(size_t iterate, Time t, PlayerIndex player) const {
        return alpha(iterate, TimeToIndex(t), player);
    }

    // Get index corresponding to the time step immediately before the given time.
    size_t TimeToIndex(Time t) const {
        return static_cast<size_t>(
                std::max<Time>(constants::kSmallNumber, t - InitialTime()) /
                time::kTimeStep);
    }

    // Get time stamp corresponding to a particular index.
    Time IndexToTime(size_t idx) const {
        return InitialTime() + time::kTimeStep * static_cast<Time>(idx);
    }

    // Save to disk.
    bool Save(bool only_last_trajectory = false,
            const std::string& experiment_name = DefaultExperimentName(), bool is_test_dir=false) const;

 private:
    // Operating points, strategies, total costs, and cumulative runtime indexed
    // by solver iterate.
    std::vector<OperatingPointDouble> operating_points_;
    std::vector<std::vector<StrategyDouble>> strategies_;
    std::vector<std::vector<double>> total_player_costs_;
    std::vector<Time> cumulative_runtimes_;
    std::vector<bool> was_converged_;
};    // class SolverLogDouble

// Utility to save a list of logs.
bool SaveLogs(const std::vector<SolverLogDouble>& logs,
            bool only_last_trajectory = true,
            const std::string& experiment_name = DefaultExperimentName());
bool SaveLogs(const std::vector<std::shared_ptr<const SolverLogDouble>>& logs,
            bool only_last_trajectory = true,
            const std::string& experiment_name = DefaultExperimentName());

}    // namespace ilqgames

#endif
