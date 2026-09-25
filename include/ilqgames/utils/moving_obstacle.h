#pragma once
#include <Eigen/Core>
#include <cmath>
#include <algorithm>
#include <utility>

namespace ilqgames {

struct MovingObs {
    Eigen::Vector3d start_pos;
    Eigen::Vector3d end_pos;
    Eigen::Vector3d size; // Dimensions
    double speed;         // m/s

    // Calculate position at time t (Looping Back-and-Forth)
    Eigen::Vector3d GetPosition(double t) const {
        if (speed <= 1e-6) return start_pos;

        double dist = (end_pos - start_pos).norm();
        if (dist < 1e-6) return start_pos;

        double duration_one_way = dist / speed;
        double cycle_duration = 2.0 * duration_one_way;

        // Modulo time to get looping behavior
        double local_t = std::fmod(t, cycle_duration);

        if (local_t < duration_one_way) {
            // Moving Start -> End
            double alpha = local_t / duration_one_way;
            return start_pos + alpha * (end_pos - start_pos);
        } else {
            // Moving End -> Start
            double alpha = (local_t - duration_one_way) / duration_one_way;
            return end_pos + alpha * (start_pos - end_pos);
        }
    }

    // Analytic Signed Distance and Gradient to an Axis-Aligned Box at time t
    std::pair<double, Eigen::Vector3d> GetDistanceAndGradient(const Eigen::Vector3d& point, double t) const {
        Eigen::Vector3d center = GetPosition(t);
        Eigen::Vector3d half_size = size * 0.5;
        
        // Transform point to box-local frame
        Eigen::Vector3d p_local = point - center;
        Eigen::Vector3d d = p_local.cwiseAbs() - half_size;

        // Signed Distance
        double outside_dist = d.cwiseMax(0.0).norm();
        double inside_dist = std::min(std::max(d.x(), std::max(d.y(), d.z())), 0.0);
        double dist = outside_dist + inside_dist;

        Eigen::Vector3d normal = Eigen::Vector3d::Zero();
        
        if (dist > 0) {
            // Outside
            Eigen::Vector3d d_pos = d.cwiseMax(0.0);
            normal.x() = (p_local.x() > 0) ? d_pos.x() : -d_pos.x();
            normal.y() = (p_local.y() > 0) ? d_pos.y() : -d_pos.y();
            normal.z() = (p_local.z() > 0) ? d_pos.z() : -d_pos.z();
            normal.normalize();
        } else {
            // Inside (Use axis of least penetration)
            if (d.x() > d.y() && d.x() > d.z()) {
                normal.x() = (p_local.x() > 0) ? 1.0 : -1.0;
            } else if (d.y() > d.z()) {
                normal.y() = (p_local.y() > 0) ? 1.0 : -1.0;
            } else {
                normal.z() = (p_local.z() > 0) ? 1.0 : -1.0;
            }
        }

        return {dist, normal};
    }
};

} // namespace ilqgames