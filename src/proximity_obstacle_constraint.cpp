#include <ilqgames/constraint/proximity_obstacle_constraint.h>
#include <ilqgames/utils/types.h>
#include <ilqgames/utils/robot_arm.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

double ProximityObstacleConstraint::Evaluate(const VectorXd& input) const {
    // update robot pos
    VectorXd rob_q(qDim_);
    for (Dimension i = 0; i < qDim_; i++)
        rob_q(i) = input(QIdx_[i]);
    rob_.updateRobotJoints(rob_q);

    double min_dist = std::numeric_limits<double>::infinity();
    const auto& capsules = rob_.get_capsules();
    const std::vector<size_t> ignore_vec = rob_.get_base_id_vec();

    for (const auto& cap : capsules) {
        if (std::find(ignore_vec.begin(), ignore_vec.end(), cap.parentFrame_) != ignore_vec.end()) {
            continue;
        }

        // We check 3 points per capsule: Start (p1), End (p2), and Midpoint
        std::vector<Eigen::Vector3d> points;
        points.push_back(cap.p1_world);
        points.push_back(cap.p2_world);
        points.push_back((cap.p1_world + cap.p2_world) * 0.5);

        for (const auto& p_world : points) {
            for (const auto& sdf : sdfs_) {
                if (!sdf->IsInBounds(p_world)) continue;

                double dist = sdf->GetDistance(p_world) - cap.radius_;
                if (dist < min_dist) min_dist = dist;
            }
        }
    }

    return k_ * (threshold_ - min_dist);
}

void ProximityObstacleConstraint::Quadraticize(Time t, const VectorXd& input, 
                                            MatrixXd* hess, VectorXd* grad) const {
    VectorXd rob_q(qDim_);
    for (Dimension i = 0; i < qDim_; i++) rob_q(i) = input(QIdx_[i]);
        
    Robot& rob_mutable = const_cast<Robot&>(rob_);
    rob_mutable.updateRobotJoints(rob_q);
    
    pinocchio::computeJointJacobians(rob_.get_model(), rob_mutable.get_data(), rob_q);
    pinocchio::updateFramePlacements(rob_.get_model(), rob_mutable.get_data());

    const auto& capsules = rob_.get_capsules();
    const std::vector<size_t> ignore_vec = rob_.get_base_id_vec();

    size_t closest_cap_idx = 0;
    double global_min_dist = std::numeric_limits<double>::infinity();
    bool found_any = false;

    for (size_t i = 0; i < capsules.size(); ++i) {
        if (std::find(ignore_vec.begin(), ignore_vec.end(), capsules[i].parentFrame_) != ignore_vec.end()) {
            continue;
        }

        const std::array<Eigen::Vector3d, 3> points = {
            capsules[i].p1_world,
            capsules[i].p2_world,
            (capsules[i].p1_world + capsules[i].p2_world) * 0.5
        };

        for (const auto& p_world : points) {
            for (const auto& sdf : sdfs_) {
                if (!sdf->IsInBounds(p_world)) continue;

                // Distance to surface = SDF - Radius
                double dist = sdf->GetDistance(p_world) - capsules[i].radius_;
                if (dist < global_min_dist) {
                    global_min_dist = dist;
                    closest_cap_idx = i;
                    found_any = true;
                }
            }
        }
    }

    const auto& best_cap = capsules[closest_cap_idx];
    Eigen::MatrixXd J_frame = Eigen::MatrixXd::Zero(6, qDim_);
    pinocchio::getFrameJacobian(rob_.get_model(), rob_.get_data(), 
                                best_cap.parentFrame_, pinocchio::LOCAL_WORLD_ALIGNED, J_frame);

    // Apply Base Rotation to Jacobian
    Eigen::Matrix3d base_rot = rob_.get_base().rotation();
    J_frame.topRows<3>() = base_rot * J_frame.topRows<3>();
    J_frame.bottomRows<3>() = base_rot * J_frame.bottomRows<3>();

    const auto& frame_transform = rob_.get_data().oMf[best_cap.parentFrame_];
    Eigen::Vector3d p_frame_origin = rob_.get_base().act(frame_transform.translation());

    const std::array<Eigen::Vector3d, 3> local_points = {
        best_cap.p1_local_,
        best_cap.p2_local_,
        (best_cap.p1_local_ + best_cap.p2_local_) * 0.5
    };

    for (const auto& p_local : local_points) {
        // Re-calculate World Position for this specific point
        Eigen::Vector3d p_world = rob_.get_base().act(frame_transform.act(p_local));

        for (const auto& sdf : sdfs_) {
            if (!sdf->IsInBounds(p_world)) continue;

            double dist = sdf->GetDistance(p_world);
            double limit = best_cap.radius_ + threshold_;

            if (dist < limit) {
                double violation = limit - dist; 
                
                Eigen::Vector3d n = sdf->GetGradient(p_world); 
                Eigen::Vector3d lever = p_world - p_frame_origin;
                Eigen::MatrixXd J_point = J_frame.topRows<3>(); 
                J_point.noalias() -= pinocchio::skew(lever) * J_frame.bottomRows<3>();

                VectorXd J_proj = (n.transpose() * J_point).transpose();
                VectorXd term = -k_ * violation * J_proj;
                
                for (Dimension i = 0; i < qDim_; i++) {
                    (*grad)(QIdx_[i]) += term(i);
                }

                for (Dimension i = 0; i < qDim_; i++) {
                    for (Dimension j = 0; j < qDim_; j++) {
                        (*hess)(QIdx_[i], QIdx_[j]) += k_ * J_proj(i) * J_proj(j);
                    }
                }
            }
        }
    }
}

} // namespace ilqgames