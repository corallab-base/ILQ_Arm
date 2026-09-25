#ifndef ILQGAMES_EXAMPLES_MULTI_ROBOT_EVAL_H
#define ILQGAMES_EXAMPLES_MULTI_ROBOT_EVAL_H

#include <ilqgames/solver/problem_double.h>
#include <ilqgames/solver/solver_params_double.h>
#include <ilqgames/solver/top_down_rob_renderable_problem.h>

#include <ilqgames/constraint/proximity_obstacle_constraint.h>

#include <ilqgames/utils/robot_arm.h>
#include <ilqgames/utils/moving_obstacle.h>

#include <ilqgames/dynamics/robot_arm_simple_6d.h>
#include <ilqgames/dynamics/single_player_dynamical_system_double.h>
#include <vector>
#include <memory>

namespace ilqgames {

class MultiRobotEval : public TopDownRobRenderableProblem {
public:
    MultiRobotEval(const std::vector<Eigen::VectorXd>& starts,
                   const std::vector<Eigen::VectorXd>& goals,
                   const std::vector<Eigen::Vector3d>& base_trans,
                   const std::vector<Eigen::Quaterniond>& base_quats,
                   const std::vector<Eigen::VectorXd>& box_configs = {},
                   bool is_floor_on = false, 
                   bool is_box_on = false,
                   bool is_move_on = false);
    ~MultiRobotEval() {}

    void ConstructDynamics();
    void ConstructInitialState();
    void ConstructPlayerCosts();

    std::vector<double> Qs(const VectorXd& x) const;
    std::vector<Robot*> get_robots() const;
    std::vector<std::shared_ptr<SignedDistanceField>>& get_sdf() const;
    std::vector<MovingObs>& get_moving_obs() const;

private:
    // Helpers to get state indices for a specific robot
    std::vector<Dimension> GetRobotQDims(size_t robot_idx) const;
    std::vector<Dimension> GetRobotUDims(size_t robot_idx) const;

    size_t num_robots_;
    std::vector<Eigen::VectorXd> starts_;
    std::vector<Eigen::VectorXd> goals_;
    std::vector<Eigen::Vector3d> base_trans_;
    std::vector<Eigen::Quaterniond> base_quats_;
    std::vector<Eigen::VectorXd> box_configs_;
    
    // Box config
    double box_xd_;
    double box_yd_;
    double box_zd_;
    double box_x_;
    double box_y_;
    double box_z_;

    // Moving axis & dist
    size_t move_axis_;
    double move_dist_;

    const bool is_floor_on_;
    const bool is_box_on_;
    const bool is_move_on_;

    std::vector<std::shared_ptr<SinglePlayerDynamicalSystemDouble>> players_systems_;
    
    std::vector<Robot*> robot_ptrs_;
    
    // Environment
    mutable std::vector<std::shared_ptr<SignedDistanceField>> env_;
    mutable std::vector<MovingObs> moving_obs_;
}; 

}  // namespace ilqgames

#endif