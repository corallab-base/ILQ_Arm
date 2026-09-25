#include <ilqgames/examples/multi_robot_eval.h>

#include <ilqgames/dynamics/concatenated_dynamical_system_double.h>

#include <ilqgames/solver/problem_double.h>
#include <ilqgames/solver/solver_params_double.h>

#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>
#include <ilqgames/utils/signed_distance_field.h>

#include <ilqgames/cost/final_time_cost_double.h>
#include <ilqgames/cost/quadratic_rob_cost.h>
#include <ilqgames/cost/squared_rob_cost.h>

#include <ilqgames/constraint/proximity_rob_constraint.h> // for robot to robot collision
#include <ilqgames/constraint/proximity_rob_self_constraint.h> // for self collision
#include <ilqgames/constraint/proximity_obstacle_constraint.h> // for static obstacle collision
#include <ilqgames/constraint/proximity_moving_obstacle_constraint.h> // for moving obstacle collision

#include <ilqgames/dynamics/robot_arm_simple_6d.h>

#include <math.h>
#include <memory>
#include <vector>
#include <numeric>

namespace ilqgames {

namespace {
// Robot URDF
const std::string rob_urdf = std::string(SOURCE_DIR) + std::string("/robot/ur5e/ur5e_real_gripper_no_joint.urdf");

// Thresholds
static constexpr double kCollisionConstWeight = 55.0;
static constexpr double kselfCollisionConstWeight = 100.0;
static constexpr double kEnvCollisionConstWeight = 300.0;
static constexpr double selfCollision_threshold = 0.01;
static constexpr double agentCollision_threshold = 0.03;
static constexpr double envCollision_threshold = 0.03;

// Costs
static constexpr float kStateRegularization = 0.45;
static constexpr float kControlRegularization = 0.45;

static const Eigen::VectorXd kControlCostWeight = []{
    Eigen::VectorXd v(6);
    v << 0.001, 0.001, 0.001, 0.001, 0.001, 0.001;
    return v;
}();

static constexpr float kFinalTimeWindow = 0.2;
static const Eigen::VectorXd kGoalCostWeight = []{
    Eigen::VectorXd v(6);
    v << 62.0, 62.0, 62.0, 62.0, 62.0, 62.0;
    return v;
}();

static const std::vector<Dimension> kRelQIds = {
    RobotArmSimple6D::kQ1Idx,
    RobotArmSimple6D::kQ2Idx,
    RobotArmSimple6D::kQ3Idx,
    RobotArmSimple6D::kQ4Idx,
    RobotArmSimple6D::kQ5Idx,
    RobotArmSimple6D::kQ6Idx
};

static const std::vector<Dimension> kRelQdIds = {
    RobotArmSimple6D::kQd1Idx,
    RobotArmSimple6D::kQd2Idx,
    RobotArmSimple6D::kQd3Idx,
    RobotArmSimple6D::kQd4Idx,
    RobotArmSimple6D::kQd5Idx,
    RobotArmSimple6D::kQd6Idx
};

static const std::vector<Dimension> kRelUIds = {
    RobotArmSimple6D::kQdd1Idx,
    RobotArmSimple6D::kQdd2Idx,
    RobotArmSimple6D::kQdd3Idx,
    RobotArmSimple6D::kQdd4Idx,
    RobotArmSimple6D::kQdd5Idx,
    RobotArmSimple6D::kQdd6Idx
};

}  // anonymous namespace

MultiRobotEval::MultiRobotEval(
    const std::vector<Eigen::VectorXd>& starts,
    const std::vector<Eigen::VectorXd>& goals,
    const std::vector<Eigen::Vector3d>& base_trans,
    const std::vector<Eigen::Quaterniond>& base_quats,
    const std::vector<Eigen::VectorXd>& box_configs,
    const bool is_floor_on, 
    const bool is_box_on,
    const bool is_move_on)
    : TopDownRobRenderableProblem(),
      starts_(starts), goals_(goals),
      base_trans_(base_trans), base_quats_(base_quats),
      is_floor_on_(is_floor_on), is_box_on_(is_box_on), is_move_on_(is_move_on),
      box_configs_(box_configs) {

    num_robots_ = starts_.size();

    // Initialize players
    for (size_t i = 0; i < num_robots_; ++i) {
        std::string name = "R" + std::to_string(i+1);
        auto p = std::make_shared<RobotArmSimple6D>(
            i, name, rob_urdf, base_trans_[i], base_quats_[i], starts_[i]
        );
        players_systems_.push_back(p);
        robot_ptrs_.push_back(&p->get_robot());
    }
}

void MultiRobotEval::ConstructDynamics() {
    dynamics_.reset(new ConcatenatedDynamicalSystemDouble(players_systems_));
}

void MultiRobotEval::ConstructInitialState() {
    x0_ = VectorXd::Zero(dynamics_->XDim());
    
    size_t current_x_offset = 0;

    for (size_t i = 0; i < num_robots_; ++i) {
        // Set Positions (Q)
        for (size_t d = 0; d < kRelQIds.size(); ++d) {
            x0_(current_x_offset + kRelQIds[d]) = starts_[i](d);
        }

        // Set Velocities (Qdot) to 0.0
        for (size_t d = 0; d < kRelQdIds.size(); ++d) {
            x0_(current_x_offset + kRelQdIds[d]) = 0.00;
        }

        current_x_offset += players_systems_[i]->XDim();
    }
}

std::vector<Dimension> MultiRobotEval::GetRobotQDims(size_t robot_idx) const {
    std::vector<Dimension> dims;
    
    size_t x_base_idx = 0;
    for (size_t k = 0; k < robot_idx; ++k) {
        x_base_idx += players_systems_[k]->XDim();
    }

    for (const auto& rel_idx : kRelQIds) {
        dims.push_back(x_base_idx + rel_idx);
    }
    return dims;
}

std::vector<Dimension> MultiRobotEval::GetRobotUDims(size_t robot_idx) const {
    return kRelUIds;
}

void MultiRobotEval::ConstructPlayerCosts() {
    env_.clear();
    moving_obs_.clear();
    if (is_floor_on_ || is_box_on_) {
        Eigen::Vector3d origin(-2, -2, -1);
        Eigen::Vector3d res(0.05, 0.05, 0.05); 
        Eigen::Vector3i dim(80, 80, 40);       

        if (is_floor_on_) {
            env_.push_back(std::make_shared<SignedDistanceField>(
                SignedDistanceField::CreateFloor(origin, res, dim, 0.0)
            ));
        }
        if (is_box_on_) {
            for (auto config : box_configs_) {
                if (config.size() > 0) {
                    double box_xd_ = config(0);
                    double box_yd_ = config(1);
                    double box_zd_ = config(2);
                    double box_x_ = config(3);
                    double box_y_ = config(4);
                    double box_z_ = config(5);
                    size_t move_axis_ = static_cast<size_t>(config(6));
                    double move_dist_ = config(7);

                    if (is_move_on_ && !(move_axis_ > 2 || move_dist_ == 0.0)) {
                        ilqgames::MovingObs box;
                        Eigen::Vector3d center_pos(box_x_, box_y_, box_z_);
                        Eigen::Vector3d dist = Eigen::Vector3d::Zero();
                        dist(move_axis_) = move_dist_;
                        box.start_pos = center_pos - dist/2;
                        box.end_pos = center_pos + dist/2;
                        box.size = Eigen::Vector3d(box_xd_, box_yd_, box_zd_);
                        box.speed = abs(move_dist_) / time::kTimeHorizon; // m/s
                        moving_obs_.push_back(box);
                    }
                    else {
                        Eigen::Vector3d center(box_x_, box_y_, box_z_);
                        Eigen::Vector3d size(box_xd_, box_yd_, box_zd_);
                        env_.push_back(std::make_shared<SignedDistanceField>(
                            SignedDistanceField::CreateBox(origin, res, dim, center, size)
                        ));
                    }
                }
            }
        }
    }

    // Costs per robot
    for (size_t i = 0; i < num_robots_; ++i) {
        std::string p_name = "P" + std::to_string(i+1);
        player_costs_.emplace_back(p_name, kStateRegularization, kControlRegularization);
        auto& p_cost = player_costs_[i];

        auto q_dims = GetRobotQDims(i);
        auto u_dims = GetRobotUDims(i);
        
        // Control Cost
        const auto ctr_effort = std::make_shared<SquaredRobCost>(
            kControlCostWeight, u_dims, p_name + " ctr effort"
        );
        p_cost.AddControlCost(i, ctr_effort);

        // Self Collision
        const std::shared_ptr<ProximityRobSelfConstraint> self_constraint(
            new ProximityRobSelfConstraint(
                selfCollision_threshold, p_name + "SelfConstraint",
                *robot_ptrs_[i], q_dims, kselfCollisionConstWeight
            )
        );
        p_cost.AddStateConstraint(self_constraint);

        // Goal Cost (Terminal)
        const auto goal_cost = std::make_shared<FinalTimeCostDouble>(
            std::make_shared<QuadraticRobCost>(kGoalCostWeight, q_dims, goals_[i], true),
            time::kTimeHorizon - kFinalTimeWindow, p_name + " Goal Q"
        );
        p_cost.AddStateCost(goal_cost);


        // Env Collision
        if (!env_.empty()) {
            const std::shared_ptr<ProximityObstacleConstraint> env_constraint(
                new ProximityObstacleConstraint(
                    envCollision_threshold, p_name + "EnvProximityConstraint",
                    env_, *robot_ptrs_[i], q_dims,
                    kEnvCollisionConstWeight
                )
            );
            p_cost.AddStateConstraint(env_constraint);
        }

        if (!moving_obs_.empty()) {
            const std::shared_ptr<ProximityMovingObstacleConstraint> moving_constraint(
                new ProximityMovingObstacleConstraint(
                    envCollision_threshold, p_name + "MoveObsProximityConstraint",
                    moving_obs_, *robot_ptrs_[i], q_dims,
                    kEnvCollisionConstWeight
                )
            );
            p_cost.AddStateConstraint(moving_constraint);
        }
    }

    for (size_t i = 0; i < num_robots_; ++i) {
        for (size_t j = i + 1; j < num_robots_; ++j) {
            auto q_dims_i = GetRobotQDims(i);
            auto q_dims_j = GetRobotQDims(j);
            std::string name = "P" + std::to_string(i+1) + "P" + std::to_string(j+1) + "Proximity";
            
            const std::shared_ptr<ProximityRobConstraint> proximity_constraint(
                new ProximityRobConstraint(
                    agentCollision_threshold, name,
                    *robot_ptrs_[i], q_dims_i,
                    *robot_ptrs_[j], q_dims_j,
                    kCollisionConstWeight
                )
            );

            player_costs_[i].AddStateConstraint(proximity_constraint);
            player_costs_[j].AddStateConstraint(proximity_constraint);
        }
    }
}

inline std::vector<double> MultiRobotEval::Qs(const VectorXd& x) const {
    std::vector<double> all_qs;
    size_t total_q = num_robots_ * kRelQIds.size();
    all_qs.reserve(total_q);

    for (size_t i = 0; i < num_robots_; ++i) {
        auto dims = GetRobotQDims(i);
        for(auto d : dims) {
            all_qs.push_back(x(d));
        }
    }
    return all_qs;
}

std::vector<Robot*> MultiRobotEval::get_robots() const {
    return robot_ptrs_;
}

std::vector<std::shared_ptr<SignedDistanceField>>& MultiRobotEval::get_sdf() const {
    return env_;
}

std::vector<MovingObs>& MultiRobotEval::get_moving_obs() const {
    return moving_obs_;
}

}  // namespace ilqgames