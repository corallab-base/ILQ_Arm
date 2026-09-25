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
                                    std::vector<ProximityRobConstraint>& inter_rob_constraints, const double rob_threshold,
                                    std::vector<ProximityObstacleConstraint>& env_constraints, const double env_threshold) {
    double res = 100;
    for (size_t i = 1; i < res; i++) {
        VectorXd inter = (1.0 - i/res) * start + i/res * goal;

        double dists;
        for (auto& rob_const : inter_rob_constraints) {
            dists = verify_collision_rob_pair(inter, rob_const) - rob_threshold;
            if (dists >= 0.0) return true;
        }
    }

    return false;
}

int main(int argc, char** argv) {
    const size_t num_cases = 250;
    const int num_robots = 2;
    const double radius = 0.6;

    const bool is_floor_on = true;
    const bool is_box_on = true;
    const bool is_move_on = false;

    const double selfCollision_threshold = 0.01;
    const double agentCollision_threshold = 0.15;
    const double envCollision_threshold = 0.09;
    
    const size_t rob_dim = 6;
    const size_t total_dim = num_robots * rob_dim;

    // Box Config
    std::vector<VectorXd> box_configs;

    VectorXd static1_info(8);
    const double static1_xd = 0.1;
    const double static1_yd = 0.5;
    const double static1_zd = 0.6;
    const double static1_x = 0.6;
    const double static1_y = 0.0;
    const double static1_z = static1_zd/2;
    size_t static1_axis = 4;
    double static1_dist = 0.0;
    static1_info << static1_xd, static1_yd, static1_zd, static1_x, static1_y, static1_z, static_cast<double>(static1_axis), static1_dist;
    box_configs.push_back(static1_info);

    VectorXd static2_info(8);
    const double static2_xd = 0.1;
    const double static2_yd = 0.5;
    const double static2_zd = 0.6;
    const double static2_x = -0.6;
    const double static2_y = 0.0;
    const double static2_z = static2_zd/2;
    size_t static2_axis = 4;
    double static2_dist = 0.0;
    static2_info << static2_xd, static2_yd, static2_zd, static2_x, static2_y, static2_z, static_cast<double>(static2_axis), static2_dist;
    box_configs.push_back(static2_info);

    VectorXd static3_info(8);
    const double static3_xd = 0.40;
    const double static3_yd = 0.15;
    const double static3_zd = 0.40;
    const double static3_x = 0.0;
    const double static3_y = 0.0;
    const double static3_z = static3_zd/2;
    size_t static3_axis = 4;
    double static3_dist = 0.0;
    static3_info << static3_xd, static3_yd, static3_zd, static3_x, static3_y, static3_z, static_cast<double>(static3_axis), static3_dist;
    box_configs.push_back(static3_info);

    Eigen::Vector3d origin(-2, -2, -1);
    Eigen::Vector3d res(0.05, 0.05, 0.05);
    Eigen::Vector3i dim(80, 80, 40);
    
    // Robot URDF
    const std::string ur5e = std::string(SOURCE_DIR) + std::string("/robot/ur5e/ur5e_real_gripper_no_joint.urdf");

    std::vector<Robot> robots;
    std::vector<std::vector<Dimension>> robot_dims;
    
    // Format per robot: [x, y, z, qw, qx, qy, qz] (Size 7)
    std::vector<VectorXd> robot_configs;
    robots.reserve(num_robots); 

    // Configuration for placement
    double start_angle = M_PI / 2.0;
    double angle_step = (2.0 * M_PI) / num_robots; 

    for (int i = 0; i < num_robots; ++i) {
        std::vector<Dimension> dims;
        for (size_t d = 0; d < rob_dim; ++d) {
            dims.push_back(i * rob_dim + d);
        }
        robot_dims.push_back(dims);

        double angle = start_angle + (i * angle_step);
        double base_x = radius * std::cos(angle);
        double base_y = radius * std::sin(angle);
        
        double yaw_to_center = std::atan2(-base_y, -base_x);
        Eigen::AngleAxisd rotation_vector(yaw_to_center, Eigen::Vector3d::UnitZ());
        Eigen::Quaterniond base_quat(rotation_vector);

        robots.emplace_back(
            i, 
            "R" + std::to_string(i+1), 
            ur5e,
            Eigen::Vector3d(base_x, base_y, 0),
            base_quat,
            Eigen::VectorXd::Zero(rob_dim)
        );
        
        VectorXd config(7);
        config << base_x, base_y, 0.0,             // Position (3)
                  base_quat.w(), base_quat.x(),    // Quaternion (4)
                  base_quat.y(), base_quat.z();
        robot_configs.push_back(config);

        std::cout << "Initialized Robot " << i << " at (" << base_x << ", " << base_y 
                  << ") facing center." << std::endl;
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

        if (is_box_on) {
            Eigen::Vector3d center1(static1_x, static1_y, static1_z);
            Eigen::Vector3d size1(static1_xd, static1_yd, static1_zd);
            auto sdf_box1 = std::make_shared<SignedDistanceField>(
                SignedDistanceField::CreateBox(origin, res, dim, center1, size1)
            );
            env.push_back(sdf_box1);
            
            Eigen::Vector3d center2(static2_x, static2_y, static2_z);
            Eigen::Vector3d size2(static2_xd, static2_yd, static2_zd);
            auto sdf_box2 = std::make_shared<SignedDistanceField>(
                SignedDistanceField::CreateBox(origin, res, dim, center2, size2)
            );
            env.push_back(sdf_box2);

            Eigen::Vector3d center3(static3_x, static3_y, static3_z);
            Eigen::Vector3d size3(static3_xd, static3_yd, static3_zd);
            auto sdf_box3 = std::make_shared<SignedDistanceField>(
                SignedDistanceField::CreateBox(origin, res, dim, center3, size3)
            );
            env.push_back(sdf_box3);
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
                robots[j], robot_dims[j]
            );
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(0.1);
    std::uniform_int_distribution<int> uniform_dist(0, num_robots-1);
    static std::normal_distribution<double> noise_dist(0.0, 0.02); 
    bool start_self_collision = dist(gen);
    size_t trials = 0;
    int rob_selc = 0;

    // Generation Loop
    VectorXd prev_full_input; 
    size_t size = 0;
    std::cout << "Starting generation..." << std::endl;

    while (size < num_cases * 2) {
        VectorXd input = VectorXd::Random(total_dim);
        bool all_valid = true;

        if (start_self_collision) {
            if (size % 2 == 0) {
                rob_selc = uniform_dist(gen);
                input(6*rob_selc + 2) = 1.8 + noise_dist(gen);
            }
            else {
                input(6*rob_selc + 2) = -1.8 + noise_dist(gen);
            }
        }


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
            VectorXd norm_q = input.unaryExpr([](double x) {
                return std::remainder(x, 2 * M_PI);
            });
            valid_pos.push_back(norm_q);
            prev_full_input = norm_q;
            size++;
            std::cout << "Start Added: " << size << std::endl;
        } 
        else {
            bool sufficient_movement = check_linear_collision(prev_full_input, input,
                                                                inter_rob_constraints, agentCollision_threshold,
                                                                env_constraints, envCollision_threshold);

            if (sufficient_movement) {
                VectorXd norm_q = input.unaryExpr([](double x) {
                    return std::remainder(x, 2 * M_PI);
                });
                valid_pos.push_back(norm_q);
                size++;
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
                            + "_" + std::to_string(num_robots) + "rob_complex_ablation";
    ilqgames::MakeDirectory(std::string(test_name));
    
    // Joint Start/Goals
    std::string file_name_pos = test_name + "/start_goal.bin";
    saveBinary(valid_pos, file_name_pos);
    std::cout << "Saved start&goal to " << file_name_pos << std::endl;

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