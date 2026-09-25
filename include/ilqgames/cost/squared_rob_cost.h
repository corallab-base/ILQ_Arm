#ifndef ILQGAMES_COST_SQUARED_ROB_COST_H
#define ILQGAMES_COST_SQUARED_ROB_COST_H

#include <ilqgames/cost/time_invariant_rob_cost.h>
#include <ilqgames/utils/types.h>

#include <string>

namespace ilqgames {

class SquaredRobCost : public TimeInvariantRobCost {
 public:
  SquaredRobCost(VectorXd weight, std::vector<Dimension> TIdx,
                const std::string& name = "")
      : TimeInvariantRobCost(weight, name), TIdx_(TIdx) {}

  // Evaluate this cost at the current input.
  double Evaluate(const VectorXd& input) const;

  // Quadraticize this cost at the given input, and add to the running
  // sum of gradients and Hessians.
  void Quadraticize(const VectorXd& input, MatrixXd* hess,
                    VectorXd* grad) const;

 private:
  // Dimension in which to apply the quadratic cost.
  const std::vector<Dimension> TIdx_;
};  //\class SquaredRobCost

}  // namespace ilqgames

#endif
