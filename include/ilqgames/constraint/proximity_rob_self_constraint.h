#ifndef ILQGAMES_CONSTRAINT_ROB_SELF_PROXIMITY_CONSTRAINT_H
#define ILQGAMES_CONSTRAINT_ROB_SELF_PROXIMITY_CONSTRAINT_H

#include <ilqgames/constraint/time_invariant_rob_constraint.h>
#include <ilqgames/utils/types.h>
#include <ilqgames/utils/robot_arm.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

class ProximityRobSelfConstraint : public TimeInvariantRobConstraint {
public:
    ~ProximityRobSelfConstraint() {}

    ProximityRobSelfConstraint(double threshold, const std::string& name,
                            Robot& robot, const std::vector<Dimension>& QIdx,
                            const double k=500)
            : TimeInvariantRobConstraint(false, name),
            rob_(robot),
            QIdx_(QIdx),
            threshold_(threshold),
            k_(k)
    {
        qDim_ = rob_.get_qDim();
        CHECK_GE(threshold_, 0.0);
    }

    double robots_min_distance(const VectorXd& input) const;

    double Compute(const VectorXd& input) const;

    double Evaluate(const VectorXd& input) const override;

    void Quadraticize(Time t, const VectorXd& input, MatrixXd* hess, VectorXd* grad) const override;

private:
    Robot& rob_;
    const std::vector<Dimension> QIdx_;
    Dimension qDim_;

    const double threshold_;
    const double k_;
};  // namespace ProximityRobSelfConstraint

}  // namespace ilqgames

#endif
