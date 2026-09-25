#ifndef ILQGAMES_COST_ROB_COST_H
#define ILQGAMES_COST_ROB_COST_H

#include <ilqgames/utils/relative_time_tracker.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <string>

namespace ilqgames {

class RobCost : public RelativeTimeTracker {
 public:
  virtual ~RobCost() {}

  // Evaluate this cost at the current time and input.
  virtual double Evaluate(Time t, const VectorXd& input) const = 0;

  // Quadraticize this cost at the given time and input, and add to the running
  // sum of gradients and Hessians.
  virtual void Quadraticize(Time t, const VectorXd& input, MatrixXd* hess,
                            VectorXd* grad) const = 0;

 protected:
  explicit RobCost(VectorXd weight, const std::string& name)
    : RelativeTimeTracker(name), weight_(weight) {}

  // Multiplicative weight associated to this cost.
  const VectorXd weight_;
};  //\class RobCost

}  // namespace ilqgames

#endif
