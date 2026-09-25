#ifndef ILQGAMES_EXAMPLES_THREE_ROBOT_DEMO1_EXAMPLE_H
#define ILQGAMES_EXAMPLES_THREE_ROBOT_DEMO1_EXAMPLE_H

#include <ilqgames/solver/problem_double.h>
#include <ilqgames/solver/solver_params_double.h>
#include <ilqgames/solver/top_down_rob_renderable_problem.h>
#include <ilqgames/constraint/proximity_obstacle_constraint.h>
#include <ilqgames/utils/robot_arm.h>

namespace ilqgames {

class ThreeRobotDemo1 : public TopDownRobRenderableProblem {
 public:
  ~ThreeRobotDemo1() {}
  ThreeRobotDemo1() : TopDownRobRenderableProblem() {}

  // Construct dynamics, initial state, and player costs.
  void ConstructDynamics();
  void ConstructInitialState();
  void ConstructPlayerCosts();

  // Unpack x, y, heading (for each player, potentially) from a given state.
  std::vector<double> Qs(const VectorXd& x) const;
  std::vector<Robot*> get_robots() const;
  std::vector<std::shared_ptr<SignedDistanceField>>& get_sdf() const;

};  // class ThreeRobotDemo1

}  // namespace ilqgames

#endif
