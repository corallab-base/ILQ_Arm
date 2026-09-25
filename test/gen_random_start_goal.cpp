#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <cassert>
#include <vector>
#include <string>
#include <algorithm>
#include <gtest/gtest.h>
#include <random>

#include <ilqgames/constraint/proximity_rob_constraint.h>
#include <ilqgames/constraint/proximity_rob_self_constraint.h>
#include <ilqgames/constraint/proximity_obstacle_constraint.h>
#include <ilqgames/cost/rob_cost.h>

#include <ilqgames/utils/types.h>
#include <ilqgames/utils/binary_save.h>
#include <ilqgames/utils/robot_arm.h>
#include <ilqgames/utils/make_directory.h>

using namespace ilqgames;

static double verify_collision_rob_pair(const VectorXd& input, ProximityRobConstraint& cost) {
    return cost.Evaluate(input);
}

static double verify_self_collision_dist(const VectorXd& robot_state, Robot& robot) {
    robot.updateRobotJoints(robot_state);
    return robot.selfMinDistance();
}

static double verify_collision_obs(const VectorXd& input, ProximityObstacleConstraint& cost) {
    return cost.Evaluate(input);
}

static bool check_linear_collision(const VectorXd& start, const VectorXd& goal,
                                    ProximityRobConstraint& cost, const double threshold, std::vector<Robot> robots) {

    double res = 300;
    size_t num_robots = start.size() / 6;
    for (size_t i = 1; i < res; i++) {
        VectorXd inter = (1.0 - i/res) * start + i/res * goal;

        double dists = verify_collision_rob_pair(inter, cost) - threshold;

        if (dists >= 0.0) {
            return true;
        }
    }

    return false;
}

int main(int argc, char** argv) {
    // CONFIGURATION
    const size_t num_cases = 250;
    const int num_robots = 3;
    const double radius = 0.6;
    const bool is_floor_on = true;
    const bool is_box_on = false;
    const bool is_move_on = false;

    const double selfCollision_threshold = 0.02;
    const double agentCollision_threshold = 0.2;
    const double envCollision_threshold = 0.09;

    // Moving axis & dist
    size_t move_axis = 0;
    double move_dist = 0.3;

    // Box Config
    std::vector<VectorXd> box_configs;
    VectorXd box_info(8);
    const double box_xd = 0.65;
    const double box_yd = 0.65;
    const double box_zd = 0.35;
    const double box_x = 0.0;
    const double box_y = 0.0;
    const double box_z = box_zd/2;
    box_info << box_xd, box_yd, box_zd, box_x, box_y, box_z, static_cast<double>(move_axis), move_dist;
    box_configs.push_back(box_info);

    Eigen::Vector3d origin(-2, -2, -1);
    Eigen::Vector3d res(0.05, 0.05, 0.05);
    Eigen::Vector3i dim(80, 80, 40);

    const size_t rob_dim = 6;
    const size_t total_dim = num_robots * rob_dim;
    
    // Robot URDF
    const std::string ur5e = std::string(SOURCE_DIR) + std::string("/robot/ur5e/ur5e_real_gripper_no_joint.urdf");

    std::vector<Robot> robots;
    std::vector<std::vector<Dimension>> robot_dims;
    
    // Format per robot: [x, y, z, qw, qx, qy, qz] (Size 7)
    std::vector<VectorXd> robot_configs;

    VectorXd config1(7);
    config1 << 0.0, 0.0, 0.0,             // Position (3)
              1.0, 0.0,    // Quaternion (4)
              0.0, 0.0;
    robot_configs.push_back(config1);

    VectorXd config2(7);
    config2 << 0.6, 0.0, 0.0,             // Position (3)
              0.0, 0.0,    // Quaternion (4)
              0.0, 1.0;
    robot_configs.push_back(config2);
    
    VectorXd config3(7);
    config3 << -0.6, 0.0, 0.0,             // Position (3)
              1.0, 0.0,    // Quaternion (4)
              0.0, 0.0;
    robot_configs.push_back(config3);

    VectorXd config4(7);
    config4 << 0.0, 0.6, 0.0,             // Position (3)
              0.70710678, 0.0,    // Quaternion (4)
              0.0, -0.70710678;
    robot_configs.push_back(config4);
    
    VectorXd config5(7);
    config5 << 0.0, -0.6, 0.0,             // Position (3)
              0.70710678, 0.0,    // Quaternion (4)
              0.0, 0.70710678;
    robot_configs.push_back(config5);

    robots.reserve(num_robots);
    for (size_t i = 0; i < num_robots; i++) {
        std::vector<Dimension> dims;
        for (size_t d = 0; d < rob_dim; ++d) {
            dims.push_back(i * rob_dim + d);
        }
        robot_dims.push_back(dims);

        robots.emplace_back(
            i, 
            "R" + std::to_string(i+1), 
            ur5e,
            Eigen::Vector3d(robot_configs[i][0], robot_configs[i][1], robot_configs[i][2]),
            Eigen::Quaterniond(robot_configs[i][3], robot_configs[i][4], robot_configs[i][5], robot_configs[i][6]),
            Eigen::VectorXd::Zero(rob_dim)
        );
    }

    std::vector<VectorXd> valid_pos;
    std::vector<std::shared_ptr<SignedDistanceField>> env;
    std::vector<std::shared_ptr<SignedDistanceField>> start_env;
    std::vector<std::shared_ptr<SignedDistanceField>> end_env;

    if (is_floor_on) {
        env.push_back(std::make_shared<SignedDistanceField>(
            SignedDistanceField::CreateFloor(origin, res, dim, 0.0)));
    }

    if (is_box_on) {
        Eigen::Vector3d center(box_x, box_y, box_z);
        Eigen::Vector3d size(box_xd, box_yd, box_zd);
        if (is_move_on) {
            Eigen::Vector3d dist = Eigen::Vector3d::Zero();
            dist(move_axis) = move_dist;
            Eigen::Vector3d start_pos = center - dist/2;
            Eigen::Vector3d end_pos = center + dist/2;

            start_env.push_back(std::make_shared<SignedDistanceField>(
                SignedDistanceField::CreateBox(origin, res, dim, start_pos, size)));

            end_env.push_back(std::make_shared<SignedDistanceField>(
                SignedDistanceField::CreateBox(origin, res, dim, end_pos, size)));
        }
        else {
            env.push_back(std::make_shared<SignedDistanceField>(
                SignedDistanceField::CreateBox(origin, res, dim, center, size)));
        }
    }
    
    // Constraints Setup
    std::vector<ProximityObstacleConstraint> env_constraints;
    std::vector<ProximityObstacleConstraint> start_env_constraints;
    std::vector<ProximityObstacleConstraint> end_env_constraints;
    for (int i = 0; i < num_robots; ++i) {
        env_constraints.emplace_back(envCollision_threshold, "env_r" + std::to_string(i), env, robots[i], robot_dims[i]);

        if (is_move_on) {
            start_env_constraints.emplace_back(envCollision_threshold, "start_env_r" + std::to_string(i), start_env, robots[i], robot_dims[i]);
            end_env_constraints.emplace_back(envCollision_threshold, "end_env_r" + std::to_string(i), end_env, robots[i], robot_dims[i]);
        }
    }

    std::vector<ProximityRobConstraint> inter_rob_constraints;
    for (int i = 0; i < num_robots; ++i) {
        for (int j = i + 1; j < num_robots; ++j) {
            inter_rob_constraints.emplace_back(
                agentCollision_threshold, 
                "prox_r" + std::to_string(i) + "_r" + std::to_string(j),
                robots[i], robot_dims[i],
                robots[j], robot_dims[j],
                1.0
            );
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(0.1);
    std::uniform_int_distribution<int> uniform_dist(0, num_robots-1);
    static std::normal_distribution<double> noise_dist(0.0, 0.02); 

    // Generation Loop
    VectorXd prev_full_input; 
    size_t size = 0;
    std::cout << "Starting generation..." << std::endl;
    std::srand(std::time(0));

    bool start_self_collision = dist(gen);
    size_t trials = 0;
    int rob_selc = 0;
    while (size < num_cases * 2) {
        VectorXd input_sample = VectorXd::Random(total_dim); //* 3;
        VectorXd input = input_sample.unaryExpr([](double x) {
                    return std::remainder(x, 2 * M_PI);
                });

        if (start_self_collision) {
            if (size % 2 == 0) {
                rob_selc = uniform_dist(gen);
                input(6*rob_selc + 2) = 1.8 + noise_dist(gen);
            }
            else {
                input(6*rob_selc + 2) = -1.8 + noise_dist(gen);
            }
        }
        else {
            if (size % 2 == 0) {
                start_self_collision = false;
            }
        }

        
        bool all_valid = true;
        for (int i = 0; i < num_robots; ++i) {
            VectorXd q_i = input.segment(i * rob_dim, rob_dim);
            if (verify_self_collision_dist(q_i, robots[i]) < selfCollision_threshold) {
                all_valid = false; break;
            }
        }
        if (!all_valid) continue;

        for (auto& env_const : env_constraints) {
            if (verify_collision_obs(input, env_const) >= 0.0) { 
                 all_valid = false; break;
            }
        }
        if (!all_valid) continue;

        for (auto& rob_const : inter_rob_constraints) {
            if (verify_collision_rob_pair(input, rob_const) > 0.0) { 
                all_valid = false; break;
            }
        }
        if (!all_valid) continue;

        if (is_move_on) {
            if (size % 2 == 0) {
                for (auto& env_const : start_env_constraints) {
                    if (verify_collision_obs(input, env_const) >= 0.0) { 
                        all_valid = false;
                        break;
                    }
                }
            }
            else {
                for (auto& env_const : end_env_constraints) {
                    if (verify_collision_obs(input, env_const) >= 0.0) { 
                        all_valid = false;
                        break;
                    }
                }
            }
        }
        if (!all_valid) continue;

        if (size % 2 == 0) {
            valid_pos.push_back(input);
            prev_full_input = input;
            size++;
            std::cout << "Start Added: " << size << std::endl;
        } 
        else {
            bool sufficient_movement = false;
            for (auto& rob_const : inter_rob_constraints) {
                bool collision = check_linear_collision(prev_full_input, input, rob_const, agentCollision_threshold, robots);
                if (collision) {
                    sufficient_movement = true;
                    break;
                }
            }
            if (sufficient_movement) {
                valid_pos.push_back(input);
                size++;
                trials = 0;
                std::cout << "Goal Added: " << size << std::endl;
                start_self_collision = dist(gen);
            }
            else {
                trials++;
                if (trials > 100) {
                    trials = 0;
                    valid_pos.pop_back();
                    size--;
                }
            }
        }
    }

    // Save
    ilqgames::MakeDirectory(std::string(ILQGAMES_TEST_DIR));
    std::string test_name = std::string(ILQGAMES_TEST_DIR) + "/test_cases/" + std::to_string(num_cases)
                            + "_" + std::to_string(num_robots) + "rob_test_ablation" + argv[1]
                            + (is_floor_on ? "_Wfloor" : "") + (is_box_on ? "_Wbox" : "") + (is_move_on ? "_Wmove" : "");
    ilqgames::MakeDirectory(std::string(test_name));
    
    // Joint Start/Goals
    std::string file_name_pos = test_name + "/start_goal.bin";
    saveBinary(valid_pos, file_name_pos);
    std::cout << "Saved start&goal to " << file_name_pos << std::endl;

    // Joint Start/Goals
    std::string file_name_cvs = test_name + "/start_goal.csv";
    saveToCSV(valid_pos, file_name_cvs);
    std::cout << "Saved start&goal to " << file_name_cvs << std::endl;

    // Robot Base Configs
    std::string file_rob_config = test_name + "/rob_configs.bin";
    saveBinary(robot_configs, file_rob_config);
    std::cout << "Saved robot configs to   " << file_rob_config << std::endl;

    // Env Setup Configs
    if (is_box_on) {
        std::string file_env_config = test_name + "/env_configs.bin";
        saveBinary(box_configs, file_env_config);
        std::cout << "Saved env configs to   " << file_env_config << std::endl;
    }


    return 0;
}