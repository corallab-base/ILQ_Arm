#ifndef ILQGAMES_CONSTRAINT_CONSTRAINT_DOUBLE_H
#define ILQGAMES_CONSTRAINT_CONSTRAINT_DOUBLE_H

#include <ilqgames/cost/rob_cost.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

class ConstraintDouble : public RobCost {
 public:
  virtual ~ConstraintDouble() {}

  // Check if this constraint is satisfied, and optionally return the constraint
  // value, which equals zero if the constraint is satisfied.
  bool IsSatisfied(Time t, const VectorXd& input, double* level) const {
    const double value = Evaluate(t, input);
    if (level) *level = value;

    return IsSatisfied(value);
  }
  bool IsSatisfied(double level) const {
    return (is_equality_) ? std::abs(level) <= constants::kSmallNumber
                          : level <= constants::kSmallNumber;
  }

  // Evaluate this constraint value, i.e., g(x), and the augmented Lagrangian,
  // i.e., lambda g(x) + mu g(x) g(x) / 2.
  virtual double Evaluate(Time t, const VectorXd& input) const = 0;
  double EvaluateAugmentedLagrangian(Time t, const VectorXd& input) const {
    const double g = Evaluate(t, input);
    const double lambda = lambdas_[TimeIndex(t)];
    return lambda * g + 0.5 * Mu(lambda, g) * g * g;
  }

  // Quadraticize the constraint value and its square, each scaled by lambda or
  // mu, respectively (terms in the augmented Lagrangian).
  virtual void Quadraticize(Time t, const VectorXd& input, MatrixXd* hess,
                            VectorXd* grad) const = 0;

  // Accessors and setters.
  bool IsEquality() const { return is_equality_; }
  double& Lambda(Time t) { return lambdas_[TimeIndex(t)]; }
  double Lambda(Time t) const { return lambdas_[TimeIndex(t)]; }
  void IncrementLambda(Time t, double value) {
    const size_t kk = TimeIndex(t);
    const double new_lambda = lambdas_[kk] + mu_ * value;
    lambdas_[kk] = (is_equality_) ? new_lambda : std::max(0.0d, new_lambda);
  }
  void ScaleLambdas(double scale) {
    for (auto& lambda : lambdas_) lambda *= scale;
  }
  static double& GlobalMu() { return mu_; }
  static void ScaleMu(double scale) { mu_ *= scale; }
  double Mu(Time t, const VectorXd& input) const {
    const double g = Evaluate(t, input);
    return Mu(Lambda(t), g);
  }
  double Mu(double lambda, double g) const {
    if (!is_equality_ && g <= constants::kSmallNumber &&
        std::abs(lambda) <= constants::kSmallNumber)
      return 0.0;
    return mu_;
  }

 protected:
  explicit ConstraintDouble(bool is_equality, const std::string& name)
      : RobCost(Eigen::VectorXd::Constant(1, 1000), name),
        is_equality_(is_equality),
        lambdas_(time::kNumTimeSteps, constants::kDefaultLambda) {}

  // Modify derivatives to account for the multipliers and the quadratic term in
  // the augmented Lagrangian. The inputs are the derivatives of g in the
  // appropriate variables (assumed to be arbitrary coordinates of the input,
  // here called x and y).
  void ModifyDerivatives(Time t, double g, double* dx, double* ddx,
                         double* dy = nullptr, double* ddy = nullptr,
                         double* dxdy = nullptr) const;

  // Is this an equality constraint? If not, it is an inequality constraint.
  bool is_equality_;

  // Multipliers, one per time step. Also a static augmented multiplier for an
  // augmented Lagrangian.
  std::vector<double> lambdas_;
  static double mu_;
};  //\class ConstraintDouble

}  // namespace ilqgames

#endif
