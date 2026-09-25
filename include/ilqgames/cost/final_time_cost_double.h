#ifndef ILQGAMES_COST_FINAL_TIME_COST_DOUBLE_H
#define ILQGAMES_COST_FINAL_TIME_COST_DOUBLE_H

#include <ilqgames/cost/rob_cost.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

class FinalTimeCostDouble : public RobCost {
 public:
  ~FinalTimeCostDouble() {}
  FinalTimeCostDouble(const std::shared_ptr<const RobCost>& cost,
                      Time threshold_time,
                      const std::string& name = "",
                      Time end_time=0.0)
      : RobCost(VectorXd::Zero(1), name), cost_(cost), threshold_time_(threshold_time), end_time_(end_time) {
    CHECK_NOTNULL(cost.get());
  }

  // Evaluate this cost at the current time and input.
  double Evaluate(Time t, const VectorXd& input) const {
    if (threshold_time_ < end_time_)
      return (t >= initial_time_ + threshold_time_ && t < initial_time_ + end_time_) ? cost_->Evaluate(t, input) : 0.0;

    else
      return (t >= initial_time_ + threshold_time_) ? cost_->Evaluate(t, input) : 0.0;
  }

  // Quadraticize this cost at the given time and input, and add to the running
  // sum of gradients and Hessians.
  void Quadraticize(Time t, const VectorXd& input, MatrixXd* hess,
                    VectorXd* grad) const {
    if (t < initial_time_ + threshold_time_) return;
    cost_->Quadraticize(t, input, hess, grad);
  }

 private:
  // RobCost function.
  const std::shared_ptr<const RobCost> cost_;

  // Time threshold relative to initial time after which to apply cost.
  const Time threshold_time_;
  const Time end_time_;
};  //\class RobCost

}  // namespace ilqgames

#endif
