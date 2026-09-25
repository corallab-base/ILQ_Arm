#ifndef ILQGAMES_UTILS_TEST_EVAL_RECORDER_H
#define ILQGAMES_UTILS_TEST_EVAL_RECORDER_H

#include <iostream>
#include <vector>
#include <fstream>
#include <numeric>
#include <string>
#include <algorithm>
#include <limits>
#include <memory>

#include <ilqgames/solver/problem_double.h>
#include <ilqgames/utils/solver_log_double.h>
#include <ilqgames/utils/operating_point_double.h> 

namespace ilqgames {

class TestEvalRecorder {
public:
    TestEvalRecorder(size_t num_tests, size_t max_iter, double goal_threshold = 0.3, double prox_threshold = 0.4)
        : num_tests_(num_tests),
          max_iter_(max_iter),
          goal_threshold_(goal_threshold),
          prox_threshold_(prox_threshold),
          num_success(0),
          num_miss_goal(0),
          num_collision(0),
          num_no_move(0),
          num_diverged(0) {
        runtimes.reserve(num_tests);
    }

    bool RecordMatch(size_t case_id, double runtime, 
                     const std::shared_ptr<const SolverLogDouble>& log, 
                     const std::shared_ptr<const ProblemDouble>& problem,
                     const std::vector<VectorXd>& start,
                     const std::vector<VectorXd>& goal,
                     const std::vector<Robot*> robots) {
        
        runtimes.push_back(runtime);
        bool is_converge = log->NumIterates() < max_iter_;
        bool is_safe = CheckProximityConstraint(log, problem);
        bool is_self_collision = CheckSelfCollision(log->FinalOperatingPoint().xs, problem, robots);

        // check end position with goal and start
        const VectorXd& final_x = log->FinalOperatingPoint().xs.back();
        std::vector<double> all_qs = problem->Qs(final_x);

        bool is_goal_reached = true;
        bool is_moved = true;
        int offset = 0;
        for (size_t rob_idx=0; rob_idx < robots.size(); rob_idx++) {
            int nq = robots[rob_idx]->get_qDim();
            if (offset + nq > (int)all_qs.size()) break;
            Eigen::Map<Eigen::VectorXd> q_sub(all_qs.data() + offset, nq);
            offset += nq;

            if (is_goal_reached && (q_sub - goal[rob_idx]).norm() >= goal_threshold_)
                is_goal_reached = false;

            if (is_moved && (q_sub - start[rob_idx]).norm() <= goal_threshold_)
                is_moved = false;
        }

        double length = CalcPathLength(log->FinalOperatingPoint().xs, problem, robots);
        path_lengths.push_back(length);

        if (is_converge) {
            if (is_safe && is_goal_reached && is_self_collision) {
                num_success++;
                success_cases.push_back(case_id);
                std::cout << "Success!!!!!!!" << std::endl; 
                return true;
            }
            else if (!is_self_collision) {
                self_collision_cases.push_back(case_id);
                std::cout << "Self Collision********" << std::endl; 
            }
            else if (!is_safe) { // collided
                num_collision++;
                collision_cases.push_back(case_id);
                std::cout << "Collision" << std::endl; 
            }
            else if (!is_moved) {
                num_no_move++;
                no_move_cases.push_back(case_id);
                std::cout << "Didn't move" << std::endl; 
            }
            else if (!is_goal_reached) {
                num_miss_goal++;
                miss_goal_cases.push_back(case_id);
                std::cout << "Didn't reach goal" << std::endl; 
            }
            return false;
        }
        else {
            num_diverged++;
            diverged_cases.push_back(case_id);
            std::cout << "Didn't converge" << std::endl; 
        }

        return false;
    }

    void SaveSummary(const std::string& test_name) const {
        const std::string dir = std::string(ILQGAMES_TEST_DIR) + "/" + test_name + "/";
        std::ofstream summary(dir + "summary.txt");
        if (summary.is_open()) {
            summary << "================ EVALUATION SUMMARY ================\n";
            summary << "Total Tests:  " << num_tests_ << "\n";
            summary << "Successes:    " << num_success << "\n";
            summary << "Success Rate: " << (num_tests_ > 0 ? (double(num_success) / num_tests_) * 100.0 : 0.0) << "%\n";
            summary << "Collisions:   " << num_collision << "\n";
            summary << "Self Collisions:   " << self_collision_cases.size() << "\n";
            summary << "Missed Goal:  " << num_miss_goal << "\n";
            summary << "No Move:      " << num_no_move << "\n";
            summary << "Converged:    " << (num_tests_ - num_diverged) << "\n";
            summary << "Diverged:     " << num_diverged << "\n";
            
            if (!runtimes.empty()) {
                double sum = std::accumulate(runtimes.begin(), runtimes.end(), 0.0);
                summary << "Avg Runtime:  " << sum / runtimes.size() << " s\n";
            }

            double sum_j = 0;
            for (size_t id : success_cases) {
                sum_j += path_lengths[id-1];
            }

            summary << "Avg Path Length:  " << sum_j / success_cases.size() << " rad\n";

            
            summary.close();
            std::cout << "[Recorder] Saved summary to " << dir << "summary.txt" << std::endl;
        }

        auto save_list = [&](const std::vector<size_t>& list, const std::string& filename) {
            std::ofstream f(dir + filename);
            if (f.is_open()) {
                for (size_t id : list) {
                    f << id << ": " <<  path_lengths[id] << " in rad\n";
                }
                f.close();
            }
        };

        save_list(success_cases,   "success_cases.txt");
        save_list(collision_cases, "collision_cases.txt");
        save_list(self_collision_cases, "self_collision_cases.txt");
        save_list(miss_goal_cases, "miss_goal_cases.txt");
        save_list(no_move_cases,   "no_move_cases.txt");
        save_list(diverged_cases, "diverging_cases.txt");
    }

private:
    double CalcPathLength(const std::vector<VectorXd>& path,
                          const std::shared_ptr<const ProblemDouble>& problem,
                          const std::vector<Robot*> robots) {

        double sum_q = 0;
        if (path.size() == 0) return sum_q;

        for (size_t i=1; i < path.size(); i++) {
            std::vector<double> q1 = problem->Qs(path[i]);
            Eigen::Map<Eigen::VectorXd> q1_vec(q1.data(), q1.size());
            std::vector<double> q2 = problem->Qs(path[i-1]);
            Eigen::Map<Eigen::VectorXd> q2_vec(q2.data(), q2.size());
            
            sum_q += (q1_vec - q2_vec).cwiseAbs().sum();
        }

        return sum_q;
    }

    bool CheckProximityConstraint(const std::shared_ptr<const SolverLogDouble>& log, 
                                  const std::shared_ptr<const ProblemDouble>& problem) const {
        
        ilqgames::PlayerCostCacheDouble cache(log, problem->PlayerCosts());
        
        size_t last_iterate = log->NumIterates() - 1;
        
        double global_max_prox = -std::numeric_limits<double>::infinity();
        bool found_prox = false;

        for (PlayerIndex p = 0; p < cache.NumPlayers(); ++p) {
            for (const auto& entry : cache.EvaluatedCosts(p)) {
                std::string cost_name = entry.first;

                if (cost_name.find("Proximity") != std::string::npos) {
                    found_prox = true;

                    const std::vector<double>& values = cache.EvaluatedCost(last_iterate, p, cost_name);
                    float max_val = -std::numeric_limits<float>::max();
                    for (double v : values) {
                        if (v > max_val) max_val = (float)v;
                    }

                    // Update global max
                    if (max_val > global_max_prox) {
                        global_max_prox = max_val;
                    }
                }
            }
        }

        if (!found_prox) return true;

        return global_max_prox < prox_threshold_;
    }

    bool CheckSelfCollision(const std::vector<VectorXd>& path,
                            const std::shared_ptr<const ProblemDouble>& problem,
                            const std::vector<Robot*> robots) {
        
        for (size_t i=0; i < path.size(); i++) {
            std::vector<double> q = problem->Qs(path[i]);
            Eigen::Map<Eigen::VectorXd> all_q(q.data(), q.size());
            
            size_t spit = 0;
            for (size_t rob_idx = 0; rob_idx < robots.size(); rob_idx++) {
                size_t rob_dim = robots[rob_idx]->get_qDim();
                Eigen::VectorXd input = all_q.segment(spit, rob_dim);
                spit += rob_dim;

                robots[rob_idx]->updateRobotJoints(input);
                double dist = robots[rob_idx]->selfMinDistance();
                if (dist < 0.0) {
                    return false;
                }
            }
        }
        return true;
    }

    const size_t num_tests_;
    const size_t max_iter_;
    std::vector<double> runtimes;

    size_t num_success;
    std::vector<size_t> success_cases;
    std::vector<double> path_lengths;

    size_t num_collision;
    std::vector<size_t> collision_cases;
    std::vector<size_t> self_collision_cases;

    size_t num_no_move;
    std::vector<size_t> no_move_cases;

    size_t num_miss_goal;
    std::vector<size_t> miss_goal_cases;

    size_t num_diverged;
    std::vector<size_t> diverged_cases;

    const double goal_threshold_;
    const double prox_threshold_;
};

}  // namespace ilqgames

#endif