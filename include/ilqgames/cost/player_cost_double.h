#ifndef ILQGAMES_COST_PLAYER_COST_DOUBLE_H
#define ILQGAMES_COST_PLAYER_COST_DOUBLE_H

#include <ilqgames/constraint/constraint_double.h>
#include <ilqgames/cost/rob_cost.h>
#include <ilqgames/utils/operating_point_double.h>
#include <ilqgames/utils/quadratic_cost_approximation_double.h>
#include <ilqgames/utils/types.h>

#include <unordered_map>

namespace ilqgames {

class PlayerCostDouble {
 public:
  ~PlayerCostDouble() {}

  // Provide default values for all constructor values. If num_time_steps is
  // positive use that to initialize the lambdas.
  explicit PlayerCostDouble(const std::string& name = "",
                      double state_regularization = 0.0,
                      double control_regularization = 0.0)
      : name_(name),
        state_regularization_(state_regularization),
        control_regularization_(control_regularization),
        cost_structure_(CostStructure::SUM),
        time_of_extreme_cost_(0) {}

  // Add new state and control costs for this player.
  void AddStateCost(const std::shared_ptr<RobCost>& cost);
  void AddControlCost(PlayerIndex idx, const std::shared_ptr<RobCost>& cost);

  // Add new state and control constraints. For now, they are only equality
  // constraints but later they should really be inequality constraints and
  // there should be some logic for maintaining sets of active constraints.
  void AddStateConstraint(const std::shared_ptr<ConstraintDouble>& constraint);
  void AddControlConstraint(PlayerIndex idx,
                            const std::shared_ptr<ConstraintDouble>& constraint);

  // Evaluate this cost at the current time, state, and controls, or
  // integrate over an entire trajectory. The "Offset" here indicates that
  // state costs will be evaluated at the next time step.
  double Evaluate(Time t, const VectorXd& x,
                 const std::vector<VectorXd>& us) const;
  double Evaluate(const OperatingPointDouble& op, Time time_step) const;
  double Evaluate(const OperatingPointDouble& op) const;
  double EvaluateOffset(Time t, Time next_t, const VectorXd& next_x,
                       const std::vector<VectorXd>& us) const;

  // Quadraticize this cost at the given time, time step, state, and controls.
  QuadraticCostApproximationDouble Quadraticize(
      Time t, const VectorXd& x, const std::vector<VectorXd>& us) const;

  // Return empty cost quadraticization except for control costs.
  QuadraticCostApproximationDouble QuadraticizeControlCosts(
      Time t, const VectorXd& x, const std::vector<VectorXd>& us) const;

  // Set whether this is a time-additive, max-over-time, or min-over-time cost.
  // At each specific time, all costs are accumulated with the given operation.
  enum CostStructure { SUM, MAX, MIN };
  void SetTimeAdditive() { cost_structure_ = SUM; }
  void SetMaxOverTime() { cost_structure_ = MAX; }
  void SetMinOverTime() { cost_structure_ = MIN; }
  bool IsTimeAdditive() const { return cost_structure_ == SUM; }
  bool IsMaxOverTime() const { return cost_structure_ == MAX; }
  bool IsMinOverTime() const { return cost_structure_ == MIN; }

  // Keep track of the time of extreme costs.
  size_t TimeOfExtremeCost() { return time_of_extreme_cost_; }
  void SetTimeOfExtremeCost(size_t kk) { time_of_extreme_cost_ = kk; }

  // Accessors.
  const PtrVector<RobCost>& StateCosts() const { return state_costs_; }
  const PlayerPtrMultiMap<RobCost>& ControlCosts() const { return control_costs_; }
  const PtrVector<ConstraintDouble>& StateConstraints() const {
    return state_constraints_;
  }
  const PlayerPtrMultiMap<ConstraintDouble>& ControlConstraints() const {
    return control_constraints_;
  }
  bool IsConstrained() const {
    return !state_constraints_.empty() || !control_constraints_.empty();
  }

 private:
  // Name to be used with error msgs.
  const std::string name_;

  // State costs and control costs.
  PtrVector<RobCost> state_costs_;
  PlayerPtrMultiMap<RobCost> control_costs_;

  // State and control constraints
  PtrVector<ConstraintDouble> state_constraints_;
  PlayerPtrMultiMap<ConstraintDouble> control_constraints_;

  // Regularization on costs.
  const double state_regularization_;
  const double control_regularization_;

  // Ternary variable whether this objective is time-additive, max-over-time, or
  // min-over-time.
  CostStructure cost_structure_;

  // Keep track of the time of extreme costs. This will depend upon the current
  // operating point, and it will only be meaningful if the cost structure is an
  // extremum over time.
  size_t time_of_extreme_cost_;
};  //\class PlayerCostDouble

}  // namespace ilqgames

#endif
