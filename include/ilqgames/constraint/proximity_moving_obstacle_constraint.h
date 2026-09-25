#pragma once

#include <ilqgames/constraint/time_invariant_rob_constraint.h>
#include <ilqgames/utils/moving_obstacle.h>
#include <ilqgames/utils/types.h>
#include <ilqgames/utils/robot_arm.h>
#include <vector>
#include <memory>
#include <string>

namespace ilqgames {

class ProximityMovingObstacleConstraint : public ConstraintDouble {
 public:
    ProximityMovingObstacleConstraint(double threshold, const std::string& name,
                                      const std::vector<MovingObs>& obstacles,
                                      Robot& robot, const std::vector<Dimension>& QIdx,
                                      double k = 100.0)
        : ConstraintDouble(false, name),
          rob_(robot),
          QIdx_(QIdx),
          obstacles_(obstacles),
          threshold_(threshold),
          k_(k) 
    {
        qDim_ = rob_.get_qDim();
    }

    double Evaluate(Time t, const VectorXd& input) const override;

    void Quadraticize(Time t, const VectorXd& input, MatrixXd* hess, VectorXd* grad) const override;

 private:
    Robot& rob_;
    const std::vector<Dimension> QIdx_;
    Dimension qDim_;

    std::vector<MovingObs> obstacles_;
    double threshold_;
    double k_;
};

} // namespace ilqgames