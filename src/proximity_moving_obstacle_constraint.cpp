#include <ilqgames/constraint/proximity_moving_obstacle_constraint.h>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <algorithm>
#include <array>

namespace ilqgames {

double ProximityMovingObstacleConstraint::Evaluate(Time t, const VectorXd& input) const {
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

        const std::array<Eigen::Vector3d, 3> points = {
            cap.p1_world, 
            cap.p2_world, 
            (cap.p1_world + cap.p2_world) * 0.5
        };

        for (const auto& obs : obstacles_) {
            for (const auto& p_world : points) {
                auto [sdf_dist, _] = obs.GetDistanceAndGradient(p_world, t);
                double dist = sdf_dist - cap.radius_;
                
                if (dist < min_dist) min_dist = dist;
            }
        }
    }

    return k_ * (threshold_ - min_dist);
}

void ProximityMovingObstacleConstraint::Quadraticize(Time t, const VectorXd& input, 
                                                     MatrixXd* hess, VectorXd* grad) const {
    VectorXd rob_q(qDim_);
    for (Dimension i = 0; i < qDim_; i++) 
        rob_q(i) = input(QIdx_[i]);
        
    Robot& rob_mutable = const_cast<Robot&>(rob_);
    rob_mutable.updateRobotJoints(rob_q);
    
    pinocchio::computeJointJacobians(rob_.get_model(), rob_mutable.get_data(), rob_q);
    pinocchio::updateFramePlacements(rob_.get_model(), rob_mutable.get_data());

    const auto& capsules = rob_.get_capsules();
    const std::vector<size_t> ignore_vec = rob_.get_base_id_vec();
    
    double min_dist = std::numeric_limits<double>::infinity();
    size_t best_cap_idx = 0;
    const MovingObs* best_obs = nullptr;
    Eigen::Vector3d best_point_world;
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

        for (const auto& obs : obstacles_) {
            for (const auto& p_world : points) {
                // Optimization: Just get distance
                auto [sdf_dist, _] = obs.GetDistanceAndGradient(p_world, t);
                double dist = sdf_dist - capsules[i].radius_;

                if (dist < min_dist) {
                    min_dist = dist;
                    best_cap_idx = i;
                    best_obs = &obs;
                    best_point_world = p_world;
                    found_any = true;
                }
            }
        }
    }

    const auto& cap = capsules[best_cap_idx];
    double violation = threshold_ - min_dist;

    auto [_, n] = best_obs->GetDistanceAndGradient(best_point_world, t);

    Eigen::MatrixXd J_frame = Eigen::MatrixXd::Zero(6, qDim_);
    pinocchio::getFrameJacobian(rob_.get_model(), rob_.get_data(), 
                                cap.parentFrame_, pinocchio::LOCAL_WORLD_ALIGNED, J_frame);
    
    Eigen::Matrix3d base_rot = rob_.get_base().rotation();
    J_frame.topRows<3>() = base_rot * J_frame.topRows<3>();
    J_frame.bottomRows<3>() = base_rot * J_frame.bottomRows<3>();
    
    Eigen::Vector3d p_frame_origin = rob_.get_base().act(
        rob_.get_data().oMf[cap.parentFrame_].translation());
    Eigen::Vector3d lever = best_point_world - p_frame_origin;

    Eigen::MatrixXd J_point = J_frame.topRows<3>();
    J_point.noalias() -= pinocchio::skew(lever) * J_frame.bottomRows<3>();

    VectorXd J_proj = J_point.transpose() * n; 
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

} // namespace ilqgames