#include <ilqgames/constraint/proximity_rob_constraint.h>
#include <ilqgames/utils/types.h>
#include <ilqgames/utils/robot_arm.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

struct ClosestPairInfo {
    CollisionOutput col;
    int frame1_idx;
    int frame2_idx;
};

double ProximityRobConstraint::robots_min_distance(const Robot& A, const Robot& B) const {
    std::vector<Capsule> capsules1 = rob1_.get_capsules();
    std::vector<Capsule> capsules2 = rob2_.get_capsules();

    double min_dist = std::numeric_limits<double>::infinity();
    for (size_t c1Idx = 0; c1Idx < capsules1.size(); c1Idx++) {
        for (size_t c2Idx = 0; c2Idx < capsules2.size(); c2Idx++) {
            double dist = capsuleDistance(capsules1[c1Idx], capsules2[c2Idx]).distance;
            if (dist < min_dist) min_dist = dist;
        }
    }

    return min_dist;
}

double ProximityRobConstraint::Evaluate(const VectorXd& input) const {
    VectorXd rob1_q(q1Dim_);
    for (Dimension i = 0; i < q1Dim_; i++)
        rob1_q(i) = input(QIdx1_[i]);
    rob1_.updateRobotJoints(rob1_q);

    VectorXd rob2_q(q2Dim_);
    for (Dimension i = 0; i < q2Dim_; i++)
        rob2_q(i) = input(QIdx2_[i]);
    rob2_.updateRobotJoints(rob2_q);

    return k_ * (threshold_ - robots_min_distance(rob1_, rob2_));
}

void ProximityRobConstraint::Quadraticize(Time t, const VectorXd& input, MatrixXd* hess, VectorXd* grad) const {
    VectorXd rob1_q(q1Dim_);
    for (Dimension i = 0; i < q1Dim_; ++i) rob1_q(i) = input(QIdx1_[i]);
    const_cast<Robot&>(rob1_).updateRobotJoints(rob1_q);
    pinocchio::computeJointJacobians(rob1_.get_model(), const_cast<Robot&>(rob1_).get_data(), rob1_q);

    VectorXd rob2_q(q2Dim_);
    for (Dimension i = 0; i < q2Dim_; ++i) rob2_q(i) = input(QIdx2_[i]);
    const_cast<Robot&>(rob2_).updateRobotJoints(rob2_q);
    pinocchio::computeJointJacobians(rob2_.get_model(), const_cast<Robot&>(rob2_).get_data(), rob2_q);

    const std::vector<Capsule>& caps1 = rob1_.get_capsules();
    const std::vector<Capsule>& caps2 = rob2_.get_capsules();

    ClosestPairInfo best = {};
    best.col.distance = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < caps1.size(); ++i) {
        for (size_t j = 0; j < caps2.size(); ++j) {
            CollisionOutput res = capsuleDistance(caps1[i], caps2[j]);
            if (res.distance < best.col.distance) {
                best.col = res;
                best.frame1_idx = caps1[i].parentFrame_;
                best.frame2_idx = caps2[j].parentFrame_;
            }
        }
    }

    double dist = best.col.distance;
    double violation = k_ * (threshold_ - dist);

    int t_idx = TimeIndex(t); 
    double lambda = lambdas_[t_idx];
    double mu = Mu(lambda, violation); // This helper handles inequality logic

    Eigen::Vector3d n = best.col.normal; // Points 2 -> 1
    VectorXd geom_grad = VectorXd::Zero(grad->size());

    // rob1 jacobian
    {
        Eigen::MatrixXd J_local = Eigen::MatrixXd::Zero(6, q1Dim_);
        pinocchio::getFrameJacobian(rob1_.get_model(), rob1_.get_data(), 
                                    best.frame1_idx, pinocchio::LOCAL_WORLD_ALIGNED, J_local);
        
        Eigen::Matrix3d base_rot = rob1_.get_base().rotation();
        J_local.topRows<3>() = base_rot * J_local.topRows<3>();
        J_local.bottomRows<3>() = base_rot * J_local.bottomRows<3>();

        Eigen::Vector3d p_frame = rob1_.get_base().act(
            rob1_.get_data().oMf[best.frame1_idx].translation());
        Eigen::Vector3d lever = best.col.p1 - p_frame;
        
        J_local.topRows<3>() -= pinocchio::skew(lever) * J_local.bottomRows<3>();

        Eigen::VectorXd term = -n.transpose() * J_local.topRows<3>();
        for(int i=0; i<q1Dim_; ++i) geom_grad(QIdx1_[i]) = term(i);
    }

    // rob2 jacobian
    {
        Eigen::MatrixXd J_local = Eigen::MatrixXd::Zero(6, q2Dim_);
        pinocchio::getFrameJacobian(rob2_.get_model(), rob2_.get_data(), 
                                    best.frame2_idx, pinocchio::LOCAL_WORLD_ALIGNED, J_local);
        
        Eigen::Matrix3d base_rot = rob2_.get_base().rotation();
        J_local.topRows<3>() = base_rot * J_local.topRows<3>();
        J_local.bottomRows<3>() = base_rot * J_local.bottomRows<3>();

        Eigen::Vector3d p_frame = rob2_.get_base().act(
            rob2_.get_data().oMf[best.frame2_idx].translation());
        Eigen::Vector3d lever = best.col.p2 - p_frame;

        J_local.topRows<3>() -= pinocchio::skew(lever) * J_local.bottomRows<3>();

        Eigen::VectorXd term = n.transpose() * J_local.topRows<3>();
        for(int i=0; i<q2Dim_; ++i) geom_grad(QIdx2_[i]) = term(i);
    }

    double al_scale = lambda + mu * violation;
    *grad += al_scale * geom_grad;
    *hess += mu * geom_grad * geom_grad.transpose();
}
}    // namespace ilqgames
