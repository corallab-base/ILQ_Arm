#include <iostream>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <cassert>
#include <gtest/gtest.h>

#include <ilqgames/constraint/constraint.h>
#include <ilqgames/constraint/proximity_rob_constraint.h>
#include <ilqgames/constraint/proximity_rob_self_constraint.h>
#include <ilqgames/constraint/proximity_obstacle_constraint.h>


#include <ilqgames/utils/types.h>
#include <ilqgames/cost/rob_cost.h>
#include <ilqgames/utils/robot_arm.h>
#include <ilqgames/constraint/proximity_rob_constraint.h>


using namespace ilqgames;

const std::string ur5e = std::string(SOURCE_DIR) + std::string("/robot/ur5e/ur5e_real_gripper_no_joint.urdf");

Robot R1(/*id*/0, "R1", ur5e,
             /*base_trans*/ Eigen::Vector3d::Zero(),
             /*base_quat*/  Eigen::Quaterniond::Identity(),
             /*init joints*/Eigen::VectorXd::Zero(6)
        );

Robot R2(/*id*/1, "R2", ur5e,
             /*base_trans*/ Eigen::Vector3d(-0.5, 0.0, 0.0),  // far along +Z
             /*base_quat*/  Eigen::Quaterniond::Identity(),
            /*init joints*/Eigen::VectorXd::Zero(6)
        );

std::vector<Dimension> dim1 = {0, 1, 2, 3, 4, 5};
std::vector<Dimension> dim2 = {12, 13, 14, 15, 16, 17};

static double varify_collision(const VectorXd& input, ProximityRobConstraint& cost) {
    double dist = cost.Evaluate(input);
    return dist;
}

static double varify_collision_self(const VectorXd& input, ProximityRobSelfConstraint& cost) {
    double dist = cost.Evaluate(input);
    return dist;
}

static double varify_collision_obs(const VectorXd& input, ProximityObstacleConstraint& cost) {
    double dist = cost.Evaluate(input);
    return dist;
}

static double varify_self_collision_dist(const VectorXd& input, Robot& robot) {
    robot.updateRobotJoints(input);
    double dist = robot.selfMinDistance();
    return dist;
}



// Joints: [-0.18694886,  2.57798929, -0.482151,    5.38780128, -4.0164897,   0.04386009]. Collision: False
// Joints: [-0.97360586,  1.83222342, -4.35656954,  4.42725034,  3.99268165, -0.97101721]. Collision: True
// Joints: [ 0.56734425 -2.56023653  3.52631724  1.37523421  5.50947242 -2.14933461]. Collision: True
// Joints: [-3.38677077, -5.90537191, -5.45940021,  4.03304424,  1.29333856, -3.57900917]. Collision: False
// Joints: [ 4.28363072,  5.25754526, -1.34021644, -3.58105785,  0.12492057, -4.12039242]. Collision: False
// Joints: [-3.05763207,  2.19914951, -3.54966797, -4.61629427, -1.10887766, -4.47973238]. Collision: False
// Joints: [ 2.26523109 -2.78856244 -3.66639803 -6.02323117 -4.12449144  0.49476253]. Collision: True
// Joints: [-2.11205304,  3.59124469,  2.45086459,  1.90459962,  5.2946279,   1.45100736]. Collision: False
// Joints: [-2.44338266,  4.66822014,  5.18719427, -1.3724266,   1.10169506,  4.30300938]. Collision: False
// Joints: [-3.85007614,  1.58636018,  6.09815892, -4.37558731, -5.63342919,  6.19979697]. Collision: True


TEST(Self_dist1, SELFCOLLISIONTEST) {
    VectorXd q(6);
    q << -0.18694886, 2.57798929, -0.482151, 5.38780128, -4.0164897, 0.04386009;

    ProximityRobSelfConstraint cost(0.0, "r1", R1, dim1);

    double dist = varify_self_collision_dist(q, R1);
    EXPECT_TRUE(dist > 0.0) << "Got " << dist;
}

TEST(Self_dist2, SELFCOLLISIONTEST) {
    VectorXd q(6);
    q << -0.97360586,  1.83222342, -4.35656954,  4.42725034,  3.99268165, -0.97101721;
    double dist = varify_self_collision_dist(q, R1);
    EXPECT_TRUE(dist <= 0.0) << "Got " << dist;
}

TEST(Self_dist3, SELFCOLLISIONTEST) {
    VectorXd q(6);
    q <<  0.56734425, -2.56023653,  3.52631724,  1.37523421,  5.50947242, -2.14933461;
    double dist = varify_self_collision_dist(q, R1);
    EXPECT_TRUE(dist <= 0.0) << "Got " << dist;
}

TEST(Self_dist4, SELFCOLLISIONTEST) {
    VectorXd q(6);
    q <<  -3.38677077, -5.90537191, -5.45940021,  4.03304424,  1.29333856, -3.57900917;
    double dist = varify_self_collision_dist(q, R1);
    EXPECT_TRUE(dist > 0.0) << "Got " << dist;
}

TEST(Self_dist5, SELFCOLLISIONTEST) {
    VectorXd q(6);
    q <<  4.28363072,  5.25754526, -1.34021644, -3.58105785,  0.12492057, -4.12039242;
    double dist = varify_self_collision_dist(q, R1);
    EXPECT_TRUE(dist > 0.0) << "Got " << dist;
}

TEST(Self_dist6, SELFCOLLISIONTEST) {
    VectorXd q(6);
    q <<  -3.05763207,  2.19914951, -3.54966797, -4.61629427, -1.10887766, -4.47973238;
    double dist = varify_self_collision_dist(q, R1);
    EXPECT_TRUE(dist <= 0.0) << "Got " << dist;
}

TEST(Self_dist7, SELFCOLLISIONTEST) {
    VectorXd q(6);
    q <<  2.26523109, -2.78856244, -3.66639803, -6.02323117, -4.12449144,  0.49476253;
    double dist = varify_self_collision_dist(q, R1);
    EXPECT_TRUE(dist <= 0.0) << "Got " << dist;
}

TEST(Self_dist8, SELFCOLLISIONTEST) {
    VectorXd q(6);
    q <<  -2.11205304,  3.59124469,  2.45086459,  1.90459962,  5.2946279,   1.45100736;
    double dist = varify_self_collision_dist(q, R1);
    EXPECT_TRUE(dist > 0.0) << "Got " << dist;
}


TEST(Self_dist9, SELFCOLLISIONTEST) {
    VectorXd q(6);
    q <<  -2.44338266,  4.66822014,  5.18719427, -1.3724266,   1.10169506,  4.30300938;
    double dist = varify_self_collision_dist(q, R1);
    EXPECT_TRUE(dist > 0.0) << "Got " << dist;
}


TEST(Self_dist10, SELFCOLLISIONTEST) {
    VectorXd q(6);
    q <<  -3.85007614,  1.58636018,  6.09815892, -4.37558731, -5.63342919,  6.19979697;

    double dist = varify_self_collision_dist(q, R1);
    EXPECT_TRUE(dist <= 0.0) << "Got " << dist;
}

// Collision check
TEST(Collision1, AGENTCOLLISIONTEST) {
    VectorXd input = VectorXd::Zero(24); // (6 j + 6jV) * 2 robot
    
    ProximityRobConstraint cost(0.00, "test1", R1, dim1, R2, dim2);

    double prox_cost = varify_collision(input, cost);
    EXPECT_TRUE(prox_cost > 0.0) << "Got " << prox_cost; // collision
}

TEST(Collision2, AGENTCOLLISIONTEST) {
    VectorXd input = VectorXd::Zero(24); // (6 j + 6jV) * 2 robot
    input(0) = 1.5707963267948966;
    
    ProximityRobConstraint cost(0.00, "test2", R1, dim1, R2, dim2);

    double prox_cost = varify_collision(input, cost);
    EXPECT_TRUE(prox_cost > 0.0) << "Got " << prox_cost; // collision
}

TEST(Collision3, AGENTCOLLISIONTEST) {
    VectorXd input = VectorXd::Zero(24); // (6 j + 6jV) * 2 robot
    input(12) = 1.5707963267948966;
    
    ProximityRobConstraint cost(0.00, "test2", R1, dim1, R2, dim2);

    double prox_cost = varify_collision(input, cost);
    EXPECT_TRUE(prox_cost <= 1e-5) << "Got " << prox_cost; // no collision
}

// Test SDF collsion with static obj
TEST(Static_dist1, OBSCOLLICIONTEST) {
    VectorXd q(6);
    q << -0.18694886, 2.57798929, -0.482151, 5.38780128, -4.0164897, 0.04386009;

    std::vector<std::shared_ptr<SignedDistanceField>> env;
    Eigen::Vector3d origin(-2, -2, -1);
    Eigen::Vector3d res(0.05, 0.05, 0.05); // 5cm resolution
    Eigen::Vector3i dim(80, 80, 40);       // 4m x 4m x 2m grid

    auto floor = std::make_shared<SignedDistanceField>(
        SignedDistanceField::CreateFloor(origin, res, dim, 0.0)
    );
    env.push_back(floor);

    ProximityObstacleConstraint cost(0.0, "r1", env, R1, dim1, 1.0);
    double prox_cost = varify_collision_obs(q, cost);

    EXPECT_TRUE(prox_cost >= 0.0) << "Got " << prox_cost;
}