#ifndef ILQGAMES_EXAMPLES_TWO_ROBOT_DEMO1_H
#define ILQGAMES_EXAMPLES_TWO_ROBOT_DEMO1_H

#include <ilqgames/solver/problem_double.h>
#include <ilqgames/solver/solver_params_double.h>
#include <ilqgames/solver/top_down_rob_renderable_problem.h>
#include <ilqgames/constraint/proximity_obstacle_constraint.h>
#include <ilqgames/utils/robot_arm.h>
#include <ilqgames/utils/moving_obstacle.h>

namespace ilqgames {

class TwoRobotComplexEnvDemo2 : public TopDownRobRenderableProblem {
public:
    ~TwoRobotComplexEnvDemo2() {}
    TwoRobotComplexEnvDemo2() : TopDownRobRenderableProblem() {}

    // Construct dynamics, initial state, and player costs.
    void ConstructDynamics();
    void ConstructInitialState();
    void ConstructPlayerCosts();

    // Unpack x, y, heading (for each player, potentially) from a given state.
    std::vector<double> Qs(const VectorXd& x) const;
    std::vector<Robot*> get_robots() const;
    std::vector<std::shared_ptr<SignedDistanceField>>& get_sdf() const;
    std::vector<MovingObs>& get_moving_obs() const;


};    // class TwoRobotComplexEnvDemo2

}    // namespace ilqgames

#endif
