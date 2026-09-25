#ifndef ILQGAMES_CONSTRAINT_ROB_PROXIMITY_CONSTRAINT_H
#define ILQGAMES_CONSTRAINT_ROB_PROXIMITY_CONSTRAINT_H

#include <ilqgames/constraint/time_invariant_rob_constraint.h>
#include <ilqgames/utils/types.h>
#include <ilqgames/utils/robot_arm.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

class ProximityRobConstraint : public TimeInvariantRobConstraint {
public:
    ~ProximityRobConstraint() {}

    ProximityRobConstraint(double threshold, const std::string& name,
                    Robot& robot1, const std::vector<Dimension>& QIdx1,
                    Robot& robot2, const std::vector<Dimension>& QIdx2,
                    const double k=100)
        : TimeInvariantRobConstraint(false, name),
        rob1_(robot1),
        QIdx1_(QIdx1),
        rob2_(robot2),
        QIdx2_(QIdx2),
        threshold_(threshold),
        k_(k)
    {
        q1Dim_ = rob1_.get_qDim();
        q2Dim_ = rob2_.get_qDim();
        CHECK_GE(threshold_, 0.0);
    }

    // Same signatures you used in the .cpp
    double robots_min_distance(const Robot& A, const Robot& B) const;

    // Evaluate this constraint value, i.e., g(x).
    // Keep VectorXd as in your existing interfaces
    double Evaluate(const VectorXd& input) const override;

    // Quadraticize (leave as-is in your project if you use it)
    void Quadraticize(Time t, const VectorXd& input, MatrixXd* hess, VectorXd* grad) const override;

private:
    Robot& rob1_;
    const std::vector<Dimension> QIdx1_;
    Dimension q1Dim_;

    Robot& rob2_;
    const std::vector<Dimension> QIdx2_;
    Dimension q2Dim_;

    const double threshold_;
    const double k_;
}; // namespace ProximityRobConstraint

}  // namespace ilqgames

#endif
