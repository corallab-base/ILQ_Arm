
#include <ilqgames/dynamics/concatenated_dynamical_system_double.h>
#include <ilqgames/examples/three_robot_demo1.h>
#include <ilqgames/solver/problem_double.h>
#include <ilqgames/solver/solver_params_double.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

// robot include
#include <ilqgames/cost/final_time_cost_double.h> // terminal reward
#include <ilqgames/cost/quadratic_rob_cost.h> // for goal joints
#include <ilqgames/cost/squared_rob_cost.h> // for ctr effort
#include <ilqgames/constraint/proximity_rob_constraint.h> // for robot to robot collision
#include <ilqgames/constraint/proximity_rob_self_constraint.h> // for self collision
#include <ilqgames/constraint/proximity_obstacle_constraint.h> // for static obstacle collision
#include <ilqgames/constraint/proximity_moving_obstacle_constraint.h> // for moving obstacle collision
#include <ilqgames/dynamics/robot_arm_simple_6d.h>

#include <math.h>
#include <memory>
#include <vector>

namespace ilqgames {

namespace {
// Robot URDF
const std::string rob1_urdf = std::string(SOURCE_DIR) + std::string("/robot/ur5e/ur5e_real_gripper_no_joint.urdf");
const std::string rob2_urdf = std::string(SOURCE_DIR) + std::string("/robot/ur5e/ur5e_real_gripper_no_joint.urdf");
const std::string rob3_urdf = std::string(SOURCE_DIR) + std::string("/robot/ur5e/ur5e_real_gripper_no_joint.urdf");

// Robot Base
const Eigen::Vector3d rob1_tran = Eigen::Vector3d(0, 0.4, 0);
const Eigen::Quaterniond rob1_quat(0.707107, -0, -0, -0.707107);

const Eigen::Vector3d rob2_tran = Eigen::Vector3d(-0.34641, -0.2, 0);
const Eigen::Quaterniond rob2_quat(0.965926, 0, 0, 0.258819);

const Eigen::Vector3d rob3_tran = Eigen::Vector3d(0.34641, -0.2, 0);
const Eigen::Quaterniond rob3_quat(0.258819, 0, 0, 0.965926);

// define player
using P1 = RobotArmSimple6D;
using P2 = RobotArmSimple6D;
using P3 = RobotArmSimple6D;

// initial joints
static const Eigen::VectorXd rob1_init_q = []{
    Eigen::VectorXd v(6);
    v << -0.374367, -0.916009,  -0.39435, -0.254673, 0.981616, 0.702025;
    return v;
}();

static const Eigen::VectorXd rob2_init_q = []{
    Eigen::VectorXd v(6);
    v << -0.879212, -0.740378, 0.398059, 0.163939, 0.0825263, -0.398426;
    return v;
}();

static const Eigen::VectorXd rob3_init_q = []{
    Eigen::VectorXd v(6);
    v << 0.893589, -0.116299, -0.515981, -0.0942656, 0.461049, -0.714494;
    return v;
}();

// goal joints
static const Eigen::VectorXd rob1_goal_q = []{
    Eigen::VectorXd v(6);
    v << 0.0980781, -0.281364, -0.959006, 0.587082, -0.682079, 0.919664;
    return v;
}();

static const Eigen::VectorXd rob2_goal_q = []{
    Eigen::VectorXd v(6);
    v << 0.913773, -0.801778, 0.0828145, 0.507119, -0.152643, -0.237545;
    return v;
}();

static const Eigen::VectorXd rob3_goal_q = []{
    Eigen::VectorXd v(6);
    v << 0.769398, -0.806028, -0.389773, 0.965774,-0.0364542, 0.739339;
    return v;
}();

auto p1 = std::make_shared<P1>(0, "R1", rob1_urdf, rob1_tran, rob1_quat, rob1_init_q);
auto p2 = std::make_shared<P2>(1, "R2", rob2_urdf, rob2_tran, rob2_quat, rob2_init_q);
auto p3 = std::make_shared<P3>(2, "R3", rob3_urdf, rob3_tran, rob3_quat, rob3_init_q);

// Collision threshold
static constexpr double kCollisionConstWeight = 30.0;
static constexpr double kEnvCollisionConstWeight = 30.0;
static constexpr double selfCollision_threshold = 0.01;
static constexpr double agentCollision_threshold = 0.03;
static constexpr double envCollision_threshold = 0.03;

// Static Environment
std::vector<std::shared_ptr<SignedDistanceField>> env;
const bool is_floor_on = true;
const bool is_box_on = false;

// box location & size
const double box_xd = 0.60;
const double box_yd = 0.15;
const double box_zd = 0.35;

const double box_x = 0.0;
const double box_y = 0.0;
const double box_z = box_zd/2;


// Cost weights.
static constexpr float kStateRegularization = 0.6;
static constexpr float kControlRegularization = 0.6;
static const Eigen::VectorXd kControlCostWeight = []{
    Eigen::VectorXd v(6);
    v << 0.001, 0.001, 0.001, 0.001, 0.001, 0.001;
    return v;
}();

// Goal cost
static constexpr float kFinalTimeWindow = 0.2;
static const Eigen::VectorXd kGoalCostWeight = []{
    Eigen::VectorXd v(6);
    v << 60.0, 60.0, 60.0, 60.0, 60.0, 60.0;
    return v;
}();

// R1
static const Dimension kP1Q1Idx = P1::kQ1Idx;
static const Dimension kP1Q2Idx = P1::kQ2Idx;
static const Dimension kP1Q3Idx = P1::kQ3Idx;
static const Dimension kP1Q4Idx = P1::kQ4Idx;
static const Dimension kP1Q5Idx = P1::kQ5Idx;
static const Dimension kP1Q6Idx = P1::kQ6Idx;
static const std::vector<Dimension> kP1Qvec = {kP1Q1Idx, kP1Q2Idx, kP1Q3Idx, kP1Q4Idx, kP1Q5Idx, kP1Q6Idx};

static const Dimension kP1Qd1Idx = P1::kQd1Idx;
static const Dimension kP1Qd2Idx = P1::kQd2Idx;
static const Dimension kP1Qd3Idx = P1::kQd3Idx;
static const Dimension kP1Qd4Idx = P1::kQd4Idx;
static const Dimension kP1Qd5Idx = P1::kQd5Idx;
static const Dimension kP1Qd6Idx = P1::kQd6Idx;

// R2
static const Dimension kP2Q1Idx = P1::kNumXDims + P2::kQ1Idx;
static const Dimension kP2Q2Idx = P1::kNumXDims + P2::kQ2Idx;
static const Dimension kP2Q3Idx = P1::kNumXDims + P2::kQ3Idx;
static const Dimension kP2Q4Idx = P1::kNumXDims + P2::kQ4Idx;
static const Dimension kP2Q5Idx = P1::kNumXDims + P2::kQ5Idx;
static const Dimension kP2Q6Idx = P1::kNumXDims + P2::kQ6Idx;
static const std::vector<Dimension> kP2Qvec = {kP2Q1Idx, kP2Q2Idx, kP2Q3Idx, kP2Q4Idx, kP2Q5Idx, kP2Q6Idx};

static const Dimension kP2Qd1Idx = P1::kNumXDims + P2::kQd1Idx;
static const Dimension kP2Qd2Idx = P1::kNumXDims + P2::kQd2Idx;
static const Dimension kP2Qd3Idx = P1::kNumXDims + P2::kQd3Idx;
static const Dimension kP2Qd4Idx = P1::kNumXDims + P2::kQd4Idx;
static const Dimension kP2Qd5Idx = P1::kNumXDims + P2::kQd5Idx;
static const Dimension kP2Qd6Idx = P1::kNumXDims + P2::kQd6Idx;

// R3
static const Dimension kP3Q1Idx = P1::kNumXDims + P2::kNumXDims + P3::kQ1Idx;
static const Dimension kP3Q2Idx = P1::kNumXDims + P2::kNumXDims + P3::kQ2Idx;
static const Dimension kP3Q3Idx = P1::kNumXDims + P2::kNumXDims + P3::kQ3Idx;
static const Dimension kP3Q4Idx = P1::kNumXDims + P2::kNumXDims + P3::kQ4Idx;
static const Dimension kP3Q5Idx = P1::kNumXDims + P2::kNumXDims + P3::kQ5Idx;
static const Dimension kP3Q6Idx = P1::kNumXDims + P2::kNumXDims + P3::kQ6Idx;
static const std::vector<Dimension> kP3Qvec = {kP3Q1Idx, kP3Q2Idx, kP3Q3Idx, kP3Q4Idx, kP3Q5Idx, kP3Q6Idx};

static const Dimension kP3Qd1Idx = P1::kNumXDims + P2::kNumXDims + P3::kQd1Idx;
static const Dimension kP3Qd2Idx = P1::kNumXDims + P2::kNumXDims + P3::kQd2Idx;
static const Dimension kP3Qd3Idx = P1::kNumXDims + P2::kNumXDims + P3::kQd3Idx;
static const Dimension kP3Qd4Idx = P1::kNumXDims + P2::kNumXDims + P3::kQd4Idx;
static const Dimension kP3Qd5Idx = P1::kNumXDims + P2::kNumXDims + P3::kQd5Idx;
static const Dimension kP3Qd6Idx = P1::kNumXDims + P2::kNumXDims + P3::kQd6Idx;

// Control dimensions.
static const Dimension kP1Qdd1Idx = P1::kQdd1Idx;
static const Dimension kP1Qdd2Idx = P1::kQdd2Idx;
static const Dimension kP1Qdd3Idx = P1::kQdd3Idx;
static const Dimension kP1Qdd4Idx = P1::kQdd4Idx;
static const Dimension kP1Qdd5Idx = P1::kQdd5Idx;
static const Dimension kP1Qdd6Idx = P1::kQdd6Idx;
static const std::vector<Dimension> kP1Uvec = {kP1Qdd1Idx, kP1Qdd2Idx, kP1Qdd3Idx, kP1Qdd4Idx, kP1Qdd5Idx, kP1Qdd6Idx};

static const Dimension kP2Qdd1Idx = P2::kQdd1Idx;
static const Dimension kP2Qdd2Idx = P2::kQdd2Idx;
static const Dimension kP2Qdd3Idx = P2::kQdd3Idx;
static const Dimension kP2Qdd4Idx = P2::kQdd4Idx;
static const Dimension kP2Qdd5Idx = P2::kQdd5Idx;
static const Dimension kP2Qdd6Idx = P2::kQdd6Idx;
static const std::vector<Dimension> kP2Uvec = {kP2Qdd1Idx, kP2Qdd2Idx, kP2Qdd3Idx, kP2Qdd4Idx, kP2Qdd5Idx, kP2Qdd6Idx};

static const Dimension kP3Qdd1Idx = P3::kQdd1Idx;
static const Dimension kP3Qdd2Idx = P3::kQdd2Idx;
static const Dimension kP3Qdd3Idx = P3::kQdd3Idx;
static const Dimension kP3Qdd4Idx = P3::kQdd4Idx;
static const Dimension kP3Qdd5Idx = P3::kQdd5Idx;
static const Dimension kP3Qdd6Idx = P3::kQdd6Idx;
static const std::vector<Dimension> kP3Uvec = {kP3Qdd1Idx, kP3Qdd2Idx, kP3Qdd3Idx, kP3Qdd4Idx, kP3Qdd5Idx, kP3Qdd6Idx};
}  // anonymous namespace

void ThreeRobotDemo1::ConstructDynamics() {
    dynamics_.reset(new ConcatenatedDynamicalSystemDouble({p1, p2, p3}));
}

void ThreeRobotDemo1::ConstructInitialState() {
    x0_ = VectorXd::Zero(dynamics_->XDim());
    x0_(kP1Q1Idx) = rob1_init_q(0);
    x0_(kP1Q2Idx) = rob1_init_q(1);
    x0_(kP1Q3Idx) = rob1_init_q(2);
    x0_(kP1Q4Idx) = rob1_init_q(3);
    x0_(kP1Q5Idx) = rob1_init_q(4);
    x0_(kP1Q6Idx) = rob1_init_q(5);

    x0_(kP1Qd1Idx) = -0.00;
    x0_(kP1Qd2Idx) = -0.00;
    x0_(kP1Qd3Idx) = -0.00;
    x0_(kP1Qd4Idx) = -0.00;
    x0_(kP1Qd5Idx) = -0.00;
    x0_(kP1Qd6Idx) = -0.00;

    x0_(kP2Q1Idx) = rob2_init_q(0);
    x0_(kP2Q2Idx) = rob2_init_q(1);
    x0_(kP2Q3Idx) = rob2_init_q(2);
    x0_(kP2Q4Idx) = rob2_init_q(3);
    x0_(kP2Q5Idx) = rob2_init_q(4);
    x0_(kP2Q6Idx) = rob2_init_q(5);

    x0_(kP2Qd1Idx) = -0.00;
    x0_(kP2Qd2Idx) = -0.00;
    x0_(kP2Qd3Idx) = -0.00;
    x0_(kP2Qd4Idx) = -0.00;
    x0_(kP2Qd5Idx) = -0.00;
    x0_(kP2Qd6Idx) = -0.00;

    x0_(kP3Q1Idx) = rob3_init_q(0);
    x0_(kP3Q2Idx) = rob3_init_q(1);
    x0_(kP3Q3Idx) = rob3_init_q(2);
    x0_(kP3Q4Idx) = rob3_init_q(3);
    x0_(kP3Q5Idx) = rob3_init_q(4);
    x0_(kP3Q6Idx) = rob3_init_q(5);

    x0_(kP3Qd1Idx) = -0.00;
    x0_(kP3Qd2Idx) = -0.00;
    x0_(kP3Qd3Idx) = -0.00;
    x0_(kP3Qd4Idx) = -0.00;
    x0_(kP3Qd5Idx) = -0.00;
    x0_(kP3Qd6Idx) = -0.00;
}

void ThreeRobotDemo1::ConstructPlayerCosts() {
    // Set up costs for all players.
    player_costs_.emplace_back("P1", kStateRegularization, kControlRegularization);
    player_costs_.emplace_back("P2", kStateRegularization, kControlRegularization);
    player_costs_.emplace_back("P3", kStateRegularization, kControlRegularization);
    
    size_t p1_id = 0;
    size_t p2_id = 1;
    size_t p3_id = 2;
    auto& p1_cost = player_costs_[p1_id];
    auto& p2_cost = player_costs_[p2_id];
    auto& p3_cost = player_costs_[p3_id];

    // Penalize control effort.
    const auto p1_ctr_effort = std::make_shared<SquaredRobCost>(
        kControlCostWeight, kP1Uvec, "P1 ctr effort"
    );
    p1_cost.AddControlCost(p1_id, p1_ctr_effort);

    const auto p2_ctr_effort = std::make_shared<SquaredRobCost>(
        kControlCostWeight, kP2Uvec, "P2 ctr effort"
    );
    p2_cost.AddControlCost(p2_id, p2_ctr_effort);

    const auto p3_ctr_effort = std::make_shared<SquaredRobCost>(
        kControlCostWeight, kP3Uvec, "P3 ctr effort"
    );
    p3_cost.AddControlCost(p3_id, p3_ctr_effort);

    // Self collision-avoidance constraints
    const std::shared_ptr<ProximityRobSelfConstraint> p1_self_constraint(
        new ProximityRobSelfConstraint(
            selfCollision_threshold, "P1SelfConstraint",
            p1->get_robot(), kP1Qvec
        )
    );
    p1_cost.AddStateConstraint(p1_self_constraint);

    const std::shared_ptr<ProximityRobSelfConstraint> p2_self_constraint(
        new ProximityRobSelfConstraint(
            selfCollision_threshold, "P2SelfConstraint",
            p2->get_robot(), kP2Qvec
        )
    );
    p2_cost.AddStateConstraint(p2_self_constraint);

    const std::shared_ptr<ProximityRobSelfConstraint> p3_self_constraint(
        new ProximityRobSelfConstraint(
            selfCollision_threshold, "P3SelfConstraint",
            p3->get_robot(), kP3Qvec
        )
    );
    p3_cost.AddStateConstraint(p3_self_constraint);

    // Collision-avoidance constraints
    const std::shared_ptr<ProximityRobConstraint> p1p2_proximity_constraint(
        new ProximityRobConstraint(
            agentCollision_threshold, "P1P2ProximityConstraint",
            p1->get_robot(), kP1Qvec,
            p2->get_robot(), kP2Qvec,
            kCollisionConstWeight
        )
    );
    p1_cost.AddStateConstraint(p1p2_proximity_constraint);
    p2_cost.AddStateConstraint(p1p2_proximity_constraint);

    const std::shared_ptr<ProximityRobConstraint> p1p3_proximity_constraint(
        new ProximityRobConstraint(
            agentCollision_threshold, "P1P3ProximityConstraint",
            p1->get_robot(), kP1Qvec,
            p3->get_robot(), kP3Qvec,
            kCollisionConstWeight
        )
    );
    p1_cost.AddStateConstraint(p1p3_proximity_constraint);
    p3_cost.AddStateConstraint(p1p3_proximity_constraint);

    const std::shared_ptr<ProximityRobConstraint> p2p3_proximity_constraint(
        new ProximityRobConstraint(
            agentCollision_threshold, "P2P3ProximityConstraint",
            p2->get_robot(), kP2Qvec,
            p3->get_robot(), kP3Qvec,
            kCollisionConstWeight
        )
    );
    p2_cost.AddStateConstraint(p2p3_proximity_constraint);
    p3_cost.AddStateConstraint(p2p3_proximity_constraint);

    // Goal costs.
    const auto P1_goal_cost = std::make_shared<FinalTimeCostDouble>(
        std::make_shared<QuadraticRobCost>(kGoalCostWeight, kP1Qvec, rob1_goal_q),
        time::kTimeHorizon - kFinalTimeWindow, "P1 Goal Q"
    );
    p1_cost.AddStateCost(P1_goal_cost);

    const auto P2_goal_cost = std::make_shared<FinalTimeCostDouble>(
        std::make_shared<QuadraticRobCost>(kGoalCostWeight, kP2Qvec, rob2_goal_q),
        time::kTimeHorizon - kFinalTimeWindow, "P2 Goal Q"
    );
    p2_cost.AddStateCost(P2_goal_cost);

    const auto P3_goal_cost = std::make_shared<FinalTimeCostDouble>(
        std::make_shared<QuadraticRobCost>(kGoalCostWeight, kP3Qvec, rob3_goal_q),
        time::kTimeHorizon - kFinalTimeWindow, "P3 Goal Q"
    );
    p3_cost.AddStateCost(P3_goal_cost);
    
    // Static env porximity cost
    if (is_floor_on || is_box_on) {
        Eigen::Vector3d origin(-2, -2, -1);
        Eigen::Vector3d res(0.05, 0.05, 0.05); // 5cm resolution
        Eigen::Vector3i dim(80, 80, 40);       // 4m x 4m x 2m grid

        if (is_floor_on) {
            auto floor = std::make_shared<SignedDistanceField>(
                SignedDistanceField::CreateFloor(origin, res, dim, 0.0)
            );
            env.push_back(floor);
        }

        if (is_box_on) {
            Eigen::Vector3d center1(0.0, 0.0, 0.4);
            Eigen::Vector3d size1(0.35, 0.35, 0.1);
            auto sdf_box1 = std::make_shared<SignedDistanceField>(
                SignedDistanceField::CreateBox(origin, res, dim, center1, size1)
            );
            env.push_back(sdf_box1);
            
            Eigen::Vector3d center2(0.0, 0.0, 0.8);
            Eigen::Vector3d size2(0.35, 0.35, 0.1);
            auto sdf_box2 = std::make_shared<SignedDistanceField>(
                SignedDistanceField::CreateBox(origin, res, dim, center2, size2)
            );
            env.push_back(sdf_box2);
        }

        const std::shared_ptr<ProximityObstacleConstraint> p1_env_proximity_constraint(
            new ProximityObstacleConstraint(
                envCollision_threshold, "P1EnvProximityConstraint",
                env, p1->get_robot(), kP1Qvec,
                kEnvCollisionConstWeight
            )
        );
        p1_cost.AddStateConstraint(p1_env_proximity_constraint);

        const std::shared_ptr<ProximityObstacleConstraint> p2_env_proximity_constraint(
            new ProximityObstacleConstraint(
                envCollision_threshold, "P2EnvProximityConstraint",
                env, p2->get_robot(), kP2Qvec,
                kEnvCollisionConstWeight
            )
        );
        p2_cost.AddStateConstraint(p2_env_proximity_constraint);

        const std::shared_ptr<ProximityObstacleConstraint> p3_env_proximity_constraint(
            new ProximityObstacleConstraint(
                envCollision_threshold, "P3EnvProximityConstraint",
                env, p3->get_robot(), kP3Qvec,
                kEnvCollisionConstWeight
            )
        );
        p3_cost.AddStateConstraint(p3_env_proximity_constraint);
    }
}

inline std::vector<double> ThreeRobotDemo1::Qs(const VectorXd& x) const {
    return {
        x(kP1Q1Idx),
        x(kP1Q2Idx),
        x(kP1Q3Idx),
        x(kP1Q4Idx),
        x(kP1Q5Idx),
        x(kP1Q6Idx),

        x(kP2Q1Idx),
        x(kP2Q2Idx),
        x(kP2Q3Idx),
        x(kP2Q4Idx),
        x(kP2Q5Idx),
        x(kP2Q6Idx),

        x(kP3Q1Idx),
        x(kP3Q2Idx),
        x(kP3Q3Idx),
        x(kP3Q4Idx),
        x(kP3Q5Idx),
        x(kP3Q6Idx),
    };
}

std::vector<Robot*> ThreeRobotDemo1::get_robots() const {
    return {&p1->get_robot(), &p2->get_robot(), &p3->get_robot()};
}


std::vector<std::shared_ptr<SignedDistanceField>>& ThreeRobotDemo1::get_sdf() const {
    return env;
}

}  // namespace ilqgames
