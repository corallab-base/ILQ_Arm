#ifndef ILQGAMES_CONSTRAINT_OBSTACLE_PROXIMITY_CONSTRAINT_H
#define ILQGAMES_CONSTRAINT_OBSTACLE_PROXIMITY_CONSTRAINT_H

#include <ilqgames/constraint/time_invariant_rob_constraint.h>
#include <ilqgames/utils/types.h>
#include <ilqgames/utils/robot_arm.h>
#include <ilqgames/utils/signed_distance_field.h>


#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

class ProximityObstacleConstraint : public TimeInvariantRobConstraint {
 public:
    /**
     * @param robot Reference to the robot model.
     * @param sdfs List of Signed Distance Fields.
     * @param threshold Extra safety distance (meters) required OUTSIDE the robot skin.
     * @param k Penalty weight.
     */
    ProximityObstacleConstraint(double threshold, const std::string& name,
                             std::vector<std::shared_ptr<SignedDistanceField>> sdfs,
                             Robot& robot, const std::vector<Dimension>& QIdx,
                             double k = 100)
        : TimeInvariantRobConstraint(false, name),
            rob_(robot),
            QIdx_(QIdx),
            sdfs_(std::move(sdfs)),
            threshold_(threshold),
            k_(k) 
    {
        qDim_ = rob_.get_qDim();
        CHECK_GE(threshold_, 0.0);
    }

    void AddSDF(std::shared_ptr<SignedDistanceField> sdf) {
        sdfs_.push_back(sdf);
    }

    double Evaluate(const VectorXd& input) const override;

    void Quadraticize(Time t, const VectorXd& input, MatrixXd* hess, VectorXd* grad) const override;

private:
    Robot& rob_;
    const std::vector<Dimension> QIdx_;
    Dimension qDim_;

    std::vector<std::shared_ptr<SignedDistanceField>> sdfs_;
    const double threshold_;
    const double k_;
};

}    // namespace ilqgames

#endif
