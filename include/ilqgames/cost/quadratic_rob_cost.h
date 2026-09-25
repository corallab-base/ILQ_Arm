#ifndef ILQGAMES_COST_QUADRATIC_COST_DOUBLE_H
#define ILQGAMES_COST_QUADRATIC_COST_DOUBLE_H

#include <ilqgames/cost/time_invariant_rob_cost.h>
#include <ilqgames/utils/types.h>

#include <string>

namespace ilqgames {

class QuadraticRobCost : public TimeInvariantRobCost {
 public:
  // Construct from a multiplicative weight and the dimension in which to apply
  // the quadratic cost (difference from nominal). If dimension < 0, then
  // applies to all dimensions (i.e. ||input - nominal * ones()||^2).
  QuadraticRobCost(VectorXd weight, std::vector<Dimension> TIdx, VectorXd nominal,
                   const bool is_relative=false, const std::string& name = "")
      : TimeInvariantRobCost(weight, name), TIdx_(TIdx), nominal_(nominal), is_relative_(is_relative) {}

  // Evaluate this cost at the current input.
  double Evaluate(const VectorXd& input) const;

  // Quadraticize this cost at the given input, and add to the running
  // sum of gradients and Hessians.
  void Quadraticize(const VectorXd& input, MatrixXd* hess,
                    VectorXd* grad) const;

 private:
  // Dimension in which to apply the quadratic cost.
  const std::vector<Dimension> TIdx_;

  // Nominal value in this (or all) dimensions.
  const VectorXd nominal_;

  const bool is_relative_;
};  //\class QuadraticRobCost

}  // namespace ilqgames

#endif
