#include <ilqgames/constraint/proximity_rob_self_constraint.h>
#include <ilqgames/utils/types.h>
#include <ilqgames/utils/robot_arm.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

double ProximityRobSelfConstraint::robots_min_distance(const VectorXd& input) const {
    rob_.updateRobotJoints(input);
    return rob_.selfMinDistance();
}

double ProximityRobSelfConstraint::Compute(const VectorXd& input) const {
    // update robot pos
    VectorXd rob_q(qDim_);
    for (Dimension i = 0; i < qDim_; i++) {
        rob_q(i) = input(QIdx_[i]);
    }
    rob_.updateRobotJoints(rob_q);

    return k_ * (threshold_ - rob_.selfMinDistance());
}

double ProximityRobSelfConstraint::Evaluate(const VectorXd& input) const {
    return Compute(input);
}

void ProximityRobSelfConstraint::Quadraticize(Time t, const VectorXd& input, MatrixXd* hess, VectorXd* grad) const {
    VectorXd rob_q(qDim_);
    for (Dimension i = 0; i < qDim_; ++i) rob_q(i) = input(QIdx_[i]);

    pinocchio::computeJointJacobians(rob_.get_model(), const_cast<Robot&>(rob_).get_data(), rob_q);
    const_cast<Robot&>(rob_).updateRobotJoints(rob_q);

    const std::vector<Capsule>& capsules = rob_.get_capsules();
    const pinocchio::Model& model = rob_.get_model();

    int t_idx = TimeIndex(t);
    double lambda = (lambdas_.size() > t_idx) ? lambdas_[t_idx] : 0.0;

    bool active_collision_found = false;

    for (size_t i = 0; i < capsules.size(); ++i) {
        for (size_t j = i + 1; j < capsules.size(); ++j) {
            const auto& c1 = capsules[i];
            const auto& c2 = capsules[j];

            if (c1.parentFrame_ == c2.parentFrame_) continue; // Same link
            
            pinocchio::JointIndex j1 = model.frames[c1.parentFrame_].parent;
            pinocchio::JointIndex j2 = model.frames[c2.parentFrame_].parent;
            
            if (j1 == j2) continue; // Same joint
            if (model.parents[j1] == j2 || model.parents[j2] == j1) continue; // Parent-Child

            CollisionOutput res = capsuleDistance(c1, c2);

            if (res.distance > threshold_) {
                continue;
            }

            active_collision_found = true;

            double violation = k_ * (threshold_ - res.distance);

            double mu = Mu(lambda, violation); 

            Eigen::Vector3d n = res.normal;
            
            Eigen::MatrixXd J1 = Eigen::MatrixXd::Zero(6, qDim_);
            pinocchio::getFrameJacobian(model, rob_.get_data(), c1.parentFrame_, pinocchio::LOCAL_WORLD_ALIGNED, J1);

            Eigen::MatrixXd J2 = Eigen::MatrixXd::Zero(6, qDim_);
            pinocchio::getFrameJacobian(model, rob_.get_data(), c2.parentFrame_, pinocchio::LOCAL_WORLD_ALIGNED, J2);

            Eigen::Matrix3d base_rot = rob_.get_base().rotation();
            
            // transform J1
            J1.topRows<3>() = base_rot * J1.topRows<3>();
            J1.bottomRows<3>() = base_rot * J1.bottomRows<3>();
            Eigen::Vector3d p1_frame = rob_.get_base().act(rob_.get_data().oMf[c1.parentFrame_].translation());
            J1.topRows<3>() -= pinocchio::skew(res.p1 - p1_frame) * J1.bottomRows<3>();

            // transform J2
            J2.topRows<3>() = base_rot * J2.topRows<3>();
            J2.bottomRows<3>() = base_rot * J2.bottomRows<3>();
            Eigen::Vector3d p2_frame = rob_.get_base().act(rob_.get_data().oMf[c2.parentFrame_].translation());
            J2.topRows<3>() -= pinocchio::skew(res.p2 - p2_frame) * J2.bottomRows<3>();

            Eigen::VectorXd pair_grad_vec = n.transpose() * (J2.topRows<3>() - J1.topRows<3>());
            
            VectorXd geom_grad = VectorXd::Zero(grad->size());
            for (int d = 0; d < qDim_; ++d) {
                geom_grad(QIdx_[d]) = k_ * pair_grad_vec(d);
            }

            double al_scale = lambda + mu * violation;
            
            *grad += al_scale * geom_grad;
            *hess += mu * geom_grad * geom_grad.transpose();
        }
    }
}
}    // namespace ilqgames
