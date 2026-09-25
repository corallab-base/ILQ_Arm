#include <ilqgames/dynamics/concatenated_dynamical_system_double.h>
#include <ilqgames/examples/two_robot_complex_env_demo3.h>
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

// Robot Base
const Eigen::Vector3d rob1_tran = Eigen::Vector3d(0.0, -0.6, 0);
const Eigen::Quaterniond rob1_quat(0.70710678, 0.0, 0.0, 0.70710678);

const Eigen::Vector3d rob2_tran = Eigen::Vector3d(0.0, 0.6, 0);
const Eigen::Quaterniond rob2_quat(0.70710678, 0.0, 0.0, -0.70710678);

// Static Environment
std::vector<std::shared_ptr<SignedDistanceField>> env;
std::vector<MovingObs> moving_obs;
const bool is_floor_on = true;
const bool is_box_on = true;
const bool is_move_on = false;

// box location & size
const double box1_xd = 0.27;
const double box1_yd = 0.27;
const double box1_zd = 0.41;
const double box1_x = 0.50;
const double box1_y = 0.0;
const double box1_z = box1_zd/2;

const double box2_xd = 0.27;
const double box2_yd = 0.27;
const double box2_zd = 0.41;
const double box2_x = -0.50;
const double box2_y = 0.0;
const double box2_z = box1_zd/2;

// define player
using P1 = RobotArmSimple6D;
using P2 = RobotArmSimple6D;

static const Eigen::VectorXd rob1_init_q = []{
    Eigen::VectorXd v(6);
    v << 0.0, -M_PI/12, M_PI/12, 0.0, M_PI/2, 0;
    return v;
}();

static const Eigen::VectorXd rob2_init_q = []{
    Eigen::VectorXd v(6);
    v << M_PI/3,-M_PI/3, M_PI/3, 0, M_PI/2, 0;
    return v;
}();

// goal joints
static const Eigen::VectorXd rob1_goal_q = []{
    Eigen::VectorXd v(6);
    v << -M_PI/2.5,-M_PI/3, M_PI/3, 0, M_PI/2, 0;
    return v;
}();

static const Eigen::VectorXd rob2_goal_q = []{
    Eigen::VectorXd v(6);
    v << -M_PI/2.5,-M_PI/3, M_PI/3, 0, M_PI/2, 0;
    return v;
}();


auto p1 = std::make_shared<P1>(0, "R1", rob1_urdf, rob1_tran, rob1_quat, rob1_init_q);
auto p2 = std::make_shared<P2>(1, "R2", rob2_urdf, rob2_tran, rob2_quat, rob2_init_q);

// Collision threshold
static constexpr double kCollisionConstWeight = 10.0;
static constexpr double kselfCollisionConstWeight = 100.0;
static constexpr double kEnvCollisionConstWeight = 20.0;
static constexpr double selfCollision_threshold = 0.01;
static constexpr double agentCollision_threshold = 0.03;
static constexpr double envCollision_threshold = 0.03;

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
    v << 62.0, 62.0, 62.0, 62.0, 62.0, 62.0;
    return v;
}();

// State dimensions
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
}  // anonymous namespace

void TwoRobotComplexEnvDemo3::ConstructDynamics() {
    dynamics_.reset(new ConcatenatedDynamicalSystemDouble({p1, p2}));
}

void TwoRobotComplexEnvDemo3::ConstructInitialState() {
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
}

void TwoRobotComplexEnvDemo3::ConstructPlayerCosts() {
    // Set up costs for all players.
    player_costs_.emplace_back("P1", kStateRegularization, kControlRegularization);
    player_costs_.emplace_back("P2", kStateRegularization, kControlRegularization);
    
    size_t p1_id = 0;
    size_t p2_id = 1;
    auto& p1_cost = player_costs_[p1_id];
    auto& p2_cost = player_costs_[p2_id];

    // Penalize control effort.
    const auto p1_ctr_effort = std::make_shared<SquaredRobCost>(
        kControlCostWeight, kP1Uvec, "P1 ctr effort"
    );
    p1_cost.AddControlCost(p1_id, p1_ctr_effort);

    const auto p2_ctr_effort = std::make_shared<SquaredRobCost>(
        kControlCostWeight, kP2Uvec, "P2 ctr effort"
    );
    p2_cost.AddControlCost(p2_id, p2_ctr_effort);

    // Self collision-avoidance constraints
    const std::shared_ptr<ProximityRobSelfConstraint> p1_self_constraint(
        new ProximityRobSelfConstraint(
            selfCollision_threshold, "P1SelfConstraint",
            p1->get_robot(), kP1Qvec, kselfCollisionConstWeight
        )
    );
    p1_cost.AddStateConstraint(p1_self_constraint);

    const std::shared_ptr<ProximityRobSelfConstraint> p2_self_constraint(
        new ProximityRobSelfConstraint(
            selfCollision_threshold, "P2SelfConstraint",
            p2->get_robot(), kP2Qvec, kselfCollisionConstWeight
        )
    );
    p2_cost.AddStateConstraint(p2_self_constraint);

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
            Eigen::Vector3d center1(box1_x, box1_y, box1_z);
            Eigen::Vector3d size1(box1_xd, box1_yd, box1_zd);
            auto sdf_box1 = std::make_shared<SignedDistanceField>(
                SignedDistanceField::CreateBox(origin, res, dim, center1, size1)
            );
            env.push_back(sdf_box1);

            Eigen::Vector3d center2(box2_x, box2_y, box2_z);
            Eigen::Vector3d size2(box2_xd, box2_yd, box2_zd);
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
    }
}

inline std::vector<double> TwoRobotComplexEnvDemo3::Qs(const VectorXd& x) const {
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
    };
}

std::vector<Robot*> TwoRobotComplexEnvDemo3::get_robots() const {
    return {&p1->get_robot(), &p2->get_robot()};
}

std::vector<std::shared_ptr<SignedDistanceField>>& TwoRobotComplexEnvDemo3::get_sdf() const {
    return env;
}

std::vector<MovingObs>& TwoRobotComplexEnvDemo3::get_moving_obs() const {
    return moving_obs;
}

}  // namespace ilqgames
