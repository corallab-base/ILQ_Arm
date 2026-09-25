#include <ilqgames/utils/make_directory.h>
#include <ilqgames/utils/operating_point_double.h>
#include <ilqgames/utils/solver_log_double.h>
#include <ilqgames/utils/strategy_double.h>
#include <ilqgames/utils/types.h>
#include <ilqgames/utils/uncopyable.h>

#include <glog/logging.h>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <regex>
#include <iomanip>
#include <sstream>
#include <string>

namespace ilqgames {

VectorXd SolverLogDouble::InterpolateState(size_t iterate, Time t) const {
    const OperatingPointDouble& op = operating_points_[iterate];

    // Low and high indices between which to interpolate.
    const size_t lo = TimeToIndex(t);
    const size_t hi = std::min(lo + 1, op.xs.size() - 1);

    // Fraction of the way between lo and hi.
    const double frac = (t - IndexToTime(lo)) / time::kTimeStep;
    return (1.0 - frac) * op.xs[lo] + frac * op.xs[hi];
}

double SolverLogDouble::InterpolateState(size_t iterate, Time t, Dimension dim) const {
    const OperatingPointDouble& op = operating_points_[iterate];

    // Low and high indices between which to interpolate.
    const size_t lo = TimeToIndex(t);
    const size_t hi = std::min(lo + 1, op.xs.size() - 1);

    // Fraction of the way between lo and hi.
    const double frac = (t - IndexToTime(lo)) / time::kTimeStep;
    return (1.0 - frac) * op.xs[lo](dim) + frac * op.xs[hi](dim);
}

VectorXd SolverLogDouble::InterpolateControl(size_t iterate, Time t, PlayerIndex player) const {
    const OperatingPointDouble& op = operating_points_[iterate];

    // Low and high indices between which to interpolate.
    const size_t lo = TimeToIndex(t);
    const size_t hi = std::min(lo + 1, op.xs.size() - 1);

    // Fraction of the way between lo and hi.
    const double frac = (t - IndexToTime(lo)) / time::kTimeStep;
    return (1.0 - frac) * op.us[lo][player] + frac * op.us[hi][player];
}

double SolverLogDouble::InterpolateControl(size_t iterate, Time t, PlayerIndex player, Dimension dim) const {
    const OperatingPointDouble& op = operating_points_[iterate];

    // Low and high indices between which to interpolate.
    const size_t lo = TimeToIndex(t);
    const size_t hi = std::min(lo + 1, op.xs.size() - 1);

    // Fraction of the way between lo and hi.
    const double frac = (t - IndexToTime(lo)) / time::kTimeStep;
    return (1.0 - frac) * op.us[lo][player](dim) + frac * op.us[hi][player](dim);
}

bool SolverLogDouble::Save(bool only_last_trajectory, const std::string& experiment_name, bool is_test_dir) const {
    // Making top-level directory
    std::string dir_name;
    if (is_test_dir)
        dir_name = std::string(ILQGAMES_TEST_DIR) + "/" + experiment_name; 
    else
        dir_name = std::string(ILQGAMES_LOG_DIR) + "/" + experiment_name; 
    
    if (!MakeDirectory(dir_name)) {
        const auto now = std::chrono::system_clock::now();
        const std::time_t t_c = std::chrono::system_clock::to_time_t(now);
        std::tm ltm = *std::localtime(&t_c);
        std::ostringstream oss;
        oss << std::put_time(&ltm, "%Y-%m-%d-%H:%M:%S");
        std::string dateTimeString = oss.str();
        dir_name = std::string(ILQGAMES_LOG_DIR) + "/" + experiment_name + "_" + dateTimeString;
        MakeDirectory(dir_name);
    }

    LOG(INFO) << "Saving to directory: " << dir_name;

    size_t start = 0;
    if (only_last_trajectory) start = operating_points_.size() - 1;

    for (size_t ii = start; ii < operating_points_.size(); ii++) {
        const auto& op = operating_points_[ii];
        const std::string sub_dir_name = dir_name + "/" + std::to_string(ii);
        if (!MakeDirectory(sub_dir_name)) return false;

        // Dump initial time.
        std::ofstream file;
        file.open(sub_dir_name + "/t0.txt");
        file << op.t0 << std::endl;
        file.close();

        // Dump xs.
        file.open(sub_dir_name + "/xs.txt");
        for (const auto& x : op.xs) {
            file << x.transpose() << std::endl;
        }
        file.close();

        // Dump total costs.
        file.open(sub_dir_name + "/costs.txt");
        for (const auto& c : total_player_costs_[ii]) {
            file << c << std::endl;
        }
        file.close();

        // Dump cumulative runtimes.
        file.open(sub_dir_name + "/cumulative_runtimes.txt");
        file << cumulative_runtimes_[ii] << std::endl;
        file.close();

        // Dump us.
        std::vector<std::ofstream> files(NumPlayers());
        for (size_t jj = 0; jj < files.size(); jj++) {
            files[jj].open(sub_dir_name + "/u" + std::to_string(jj) + ".txt");
        }
        for (size_t kk = 0; kk < op.us.size(); kk++) {
            CHECK_EQ(files.size(), op.us[kk].size());
            for (size_t jj = 0; jj < files.size(); jj++) {
                files[jj] << op.us[kk][jj].transpose() << std::endl;
            }
        }
        for (size_t jj = 0; jj < files.size(); jj++) {
            files[jj].close();
        }
    }

    return true;
}

inline std::vector<MatrixXd> SolverLogDouble::Ps(size_t iterate, size_t time_index) const {
    std::vector<MatrixXd> Ps(strategies_[iterate].size());
    for (PlayerIndex ii = 0; ii < Ps.size(); ii++)
        Ps[ii] = P(iterate, time_index, ii);
    return Ps;
}

inline std::vector<VectorXd> SolverLogDouble::alphas(size_t iterate, size_t time_index) const {
    std::vector<VectorXd> alphas(strategies_[iterate].size());
    for (PlayerIndex ii = 0; ii < alphas.size(); ii++)
        alphas[ii] = alpha(iterate, time_index, ii);
    return alphas;
}

inline MatrixXd SolverLogDouble::P(size_t iterate, size_t time_index,
                                                         PlayerIndex player) const {
    return strategies_[iterate][player].Ps[time_index];
}

inline VectorXd SolverLogDouble::alpha(size_t iterate, size_t time_index,
                                                                 PlayerIndex player) const {
    return strategies_[iterate][player].alphas[time_index];
}

bool SaveLogs(const std::vector<SolverLogDouble>& logs, bool only_last_trajectory,
                            const std::string& experiment_name) {
    const std::string dir_name =
            std::string(ILQGAMES_LOG_DIR) + "/" + experiment_name;
    if (!MakeDirectory(dir_name)) return false;

    for (size_t ii = 0; ii < logs.size(); ii++) {
        const auto& log = logs[ii];

        if (!log.Save(only_last_trajectory,
                                    experiment_name + "/" + std::to_string(ii)))
            return false;
    }

    return true;
}

bool SaveLogs(const std::vector<std::shared_ptr<const SolverLogDouble>>& logs,
                            bool only_last_trajectory, const std::string& experiment_name) {
    const std::string dir_name =
            std::string(ILQGAMES_LOG_DIR) + "/" + experiment_name;
    if (!MakeDirectory(dir_name)) return false;

    for (size_t ii = 0; ii < logs.size(); ii++) {
        const auto& log = logs[ii];

        if (!log->Save(only_last_trajectory, experiment_name + "/" + std::to_string(ii)))
            return false;
    }

    return true;
}

}    // namespace ilqgames
