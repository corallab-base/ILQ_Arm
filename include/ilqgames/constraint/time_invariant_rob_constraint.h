#ifndef ILQGAMES_CONSTRAINT_TIME_INVARIANT_ROB_CONSTRAINT_H
#define ILQGAMES_CONSTRAINT_TIME_INVARIANT_ROB_CONSTRAINT_H

#include <ilqgames/constraint/constraint_double.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

class TimeInvariantRobConstraint : public ConstraintDouble {
 public:
  virtual ~TimeInvariantRobConstraint() {}

  // Evaluate this constraint value, i.e., g(x).
  virtual double Evaluate(const VectorXd& input) const = 0;
  
  double Evaluate(Time t, const VectorXd& input) const {
    return Evaluate(input);
  };

  // Quadraticize the constraint value and its square, each scaled by lambda or
  // mu, respectively (terms in the augmented Lagrangian).
  // NOTE: this is time-varying because time is used to select lambda.
  virtual void Quadraticize(Time t, const VectorXd& input, MatrixXd* hess,
                            VectorXd* grad) const = 0;

 protected:
  explicit TimeInvariantRobConstraint(bool is_equality, const std::string& name)
      : ConstraintDouble(is_equality, name) {}
};  // namespace TimeInvariantRobConstraint

}  // namespace ilqgames

#endif
