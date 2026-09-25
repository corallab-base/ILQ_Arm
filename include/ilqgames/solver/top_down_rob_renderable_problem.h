#ifndef ILQGAMES_SOLVER_TOP_DOWN_ROB_RENDERABLE_PROBLEM_H
#define ILQGAMES_SOLVER_TOP_DOWN_ROB_RENDERABLE_PROBLEM_H

#include <ilqgames/solver/problem_double.h>
#include <ilqgames/utils/types.h>

namespace ilqgames {

class TopDownRobRenderableProblem : public ProblemDouble {
 public:
  virtual ~TopDownRobRenderableProblem() {}

  // Unpack x, y, heading (for each player, potentially) from a given state.
  virtual std::vector<double> Qs(const VectorXd& x) const = 0;

 protected:
  TopDownRobRenderableProblem() : ProblemDouble() {}
};  // class TopDownRobRenderableProblem

}  // namespace ilqgames

#endif
