#include <ilqgames/cost/rob_cost.h>
#include <ilqgames/cost/player_cost_double.h>
#include <ilqgames/utils/operating_point_double.h>
#include <ilqgames/utils/quadratic_cost_approximation_double.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <unordered_map>

namespace ilqgames {

namespace {

// Accumulate control costs and constraints into the given quadratic
// approximation.
template <typename T, typename F>
void AccumulateControlCostsBase(const PlayerPtrMultiMap<T>& costs, Time t,
                                const std::vector<VectorXd>& us,
                                double regularization,
                                QuadraticCostApproximationDouble* q, F f) {
    size_t cost_idx = 0;
    for (const auto& pair : costs) {
        const PlayerIndex player = pair.first;
        const auto& cost = pair.second;

        // If we haven't seen this player yet, initialize R and r to zero.
        auto iter = q->control.find(player);
        if (iter == q->control.end()) {
            auto inserted_pair = q->control.emplace(
                    player, SingleCostApproximationDouble(us[player].size(), regularization));

            // Second element should be true because we definitely won't have any
            // key collisions.
            CHECK(inserted_pair.second);

            // Update iter to point to where the new R was inserted.
            iter = inserted_pair.first;
        }

        f(*cost, t, us[player], &(iter->second.hess), &(iter->second.grad));
        cost_idx++;
    }
}

void AccumulateControlCosts(const PlayerPtrMultiMap<RobCost>& costs, Time t,
                            const std::vector<VectorXd>& us,
                            double regularization,
                            QuadraticCostApproximationDouble* q) {
    auto f = [](const RobCost& cost, Time t, const VectorXd& u, MatrixXd* hess,
                            VectorXd* grad) { cost.Quadraticize(t, u, hess, grad); };
    AccumulateControlCostsBase(costs, t, us, regularization, q, f);
}

void AccumulateControlConstraints(
        const PlayerPtrMultiMap<ConstraintDouble>& constraints, Time t,
        const std::vector<VectorXd>& us, double regularization,
        QuadraticCostApproximationDouble* q) {
    auto f = [](const ConstraintDouble& constraint, Time t, const VectorXd& u,
                            MatrixXd* hess,
                            VectorXd* grad) { constraint.Quadraticize(t, u, hess, grad); };
    AccumulateControlCostsBase(constraints, t, us, regularization, q, f);
}

}    // namespace

void PlayerCostDouble::AddStateCost(const std::shared_ptr<RobCost>& cost) {
    state_costs_.emplace_back(cost);
}

void PlayerCostDouble::AddControlCost(PlayerIndex idx, const std::shared_ptr<RobCost>& cost) {
    control_costs_.emplace(idx, cost);
}

void PlayerCostDouble::AddStateConstraint(
        const std::shared_ptr<ConstraintDouble>& constraint) {
    state_constraints_.emplace_back(constraint);
}

void PlayerCostDouble::AddControlConstraint(
        PlayerIndex idx, const std::shared_ptr<ConstraintDouble>& constraint) {
    control_constraints_.emplace(idx, constraint);
}

double PlayerCostDouble::Evaluate(Time t, const VectorXd& x,
                                const std::vector<VectorXd>& us) const {
    double total_cost = 0.0;

    // State costs.
    for (const auto& cost : state_costs_) total_cost += cost->Evaluate(t, x);

    // Control costs.
    for (const auto& pair : control_costs_) {
        const PlayerIndex& player = pair.first;
        const auto& cost = pair.second;

        total_cost += cost->Evaluate(t, us[player]);
    }

    return total_cost;
}

double PlayerCostDouble::Evaluate(const OperatingPointDouble& op, Time time_step) const {
    double cost = 0.0;
    if (IsMinOverTime())
        cost = constants::kInfinity;
    else if (IsMaxOverTime())
        cost = -constants::kInfinity;

    for (size_t kk = 0; kk < op.xs.size(); kk++) {
        const Time t = op.t0 + time_step * static_cast<double>(kk);
        const double instantaneous_cost = Evaluate(t, op.xs[kk], op.us[kk]);

        if (IsTimeAdditive())
            cost += instantaneous_cost;
        else if (IsMinOverTime())
            cost = std::min(cost, instantaneous_cost);
        else
            cost = std::max(cost, instantaneous_cost);
    }

    return cost;
}

double PlayerCostDouble::Evaluate(const OperatingPointDouble& op) const {
    double total_cost = 0.0;
    for (size_t kk = 0; kk < op.xs.size(); kk++) total_cost += Evaluate(op, kk);

    return total_cost;
}

double PlayerCostDouble::EvaluateOffset(Time t, Time next_t, const VectorXd& next_x,
                                        const std::vector<VectorXd>& us) const {
    double total_cost = 0.0;

    // State costs.
    for (const auto& cost : state_costs_)
        total_cost += cost->Evaluate(next_t, next_x);

    // Control costs.
    for (const auto& pair : control_costs_) {
        const PlayerIndex& player = pair.first;
        const auto& cost = pair.second;

        total_cost += cost->Evaluate(t, us[player]);
    }

    return total_cost;
}

QuadraticCostApproximationDouble PlayerCostDouble::Quadraticize(
        Time t, const VectorXd& x, const std::vector<VectorXd>& us) const {
    QuadraticCostApproximationDouble q(x.size(), state_regularization_);

    // Accumulate state costs.
    for (const auto& cost : state_costs_)
        cost->Quadraticize(t, x, &q.state.hess, &q.state.grad);

    // Accumulate control costs.
    AccumulateControlCosts(control_costs_, t, us, control_regularization_, &q);

    // Accumulate state constraints (including augmented Lagrangian terms scaled
    // by appropriate multipliers).
    for (const auto& constraint : state_constraints_)
        constraint->Quadraticize(t, x, &q.state.hess, &q.state.grad);

    // Accumulate control constraints.
    AccumulateControlConstraints(control_constraints_, t, us, control_regularization_, &q);

    return q;
}

QuadraticCostApproximationDouble PlayerCostDouble::QuadraticizeControlCosts(
        Time t, const VectorXd& x, const std::vector<VectorXd>& us) const {
    QuadraticCostApproximationDouble q(x.size(), state_regularization_);

    // Accumulate control costs.
    AccumulateControlCosts(control_costs_, t, us, control_regularization_, &q);

    return q;
}

}    // namespace ilqgames
