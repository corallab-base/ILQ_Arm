#pragma once
#include <filesystem>
#include <ilqgames/utils/types.h>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/config.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <algorithm>
#include <regex>
#include <map>
#include <limits>

static constexpr double EPSILON = 1e-9;

namespace ilqgames {

struct Capsule {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    const std::string name_;
    const size_t parentFrame_;
    const Eigen::Vector3d p1_local_; // Store local coordinate relative to frame
    const Eigen::Vector3d p2_local_; // Store local coordinate relative to frame
    const double radius_;

    Eigen::Vector3d p1_world;
    Eigen::Vector3d p2_world;

    explicit Capsule(Eigen::Vector3d p1_local, Eigen::Vector3d p2_local, double radius, std::string name, int parent_idx)
    : p1_local_(p1_local), p2_local_(p2_local), radius_(radius), name_(name), parentFrame_(parent_idx) {};

    void transform(const pinocchio::SE3 &oMf) {
        p1_world = oMf.act(p1_local_);
        p2_world = oMf.act(p2_local_);
    }
};

struct CollisionOutput {
    double distance;
    Eigen::Vector3d p1; // Point on segment 1 (World Frame)
    Eigen::Vector3d p2; // Point on segment 2 (World Frame)
    Eigen::Vector3d normal; // Unit vector pointing from p2 to p1
};

inline CollisionOutput segmentSegmentDistance(
    const Eigen::Vector3d &p1, const Eigen::Vector3d &q1,
    const Eigen::Vector3d &p2, const Eigen::Vector3d &q2)
{
    const Eigen::Vector3d d1 = q1 - p1;
    const Eigen::Vector3d d2 = q2 - p2;
    const Eigen::Vector3d r = p1 - p2;

    // Squared lengths and dot products
    const double a = d1.dot(d1); 
    const double e = d2.dot(d2); 
    const double f = d2.dot(r);  
    const double c = d1.dot(r);  
    const double b = d1.dot(d2); 
    
    // Denominator for the general case
    const double denom = a * e - b * b; 

    double t1 = 0.0;
    double t2 = 0.0;

    if (a <= EPSILON && e <= EPSILON) { 
        // Both segments are points
        t1 = 0.0; t2 = 0.0;
        const Eigen::Vector3d closest_p1 = p1;
        const Eigen::Vector3d closest_p2 = p2;
        double dist = (closest_p1 - closest_p2).norm();
        
        Eigen::Vector3d normal;
        if (dist > EPSILON) {
            normal = (closest_p1 - closest_p2) / dist;
        } else {
            normal = Eigen::Vector3d::UnitZ();
        }
        return {dist, closest_p1, closest_p2, normal};
    }
    
    if (a <= EPSILON) {
        // Segment 1 is a point. Minimize |p1 - (p2 + t2*d2)|
        t1 = 0.0;
        t2 = std::clamp(f / e, 0.0, 1.0);
    } 
    else if (e <= EPSILON) {
        // Segment 2 is a point. Minimize |(p1 + t1*d1) - p2|
        t2 = 0.0;
        t1 = std::clamp(-c / a, 0.0, 1.0);
    } 
    // Non-Parallel
    else if (denom > EPSILON) {
        t1 = std::clamp((b * f - c * e) / denom, 0.0, 1.0);
        t2 = (b * t1 + f) / e;

        if (t2 < 0.0) {
            t2 = 0.0;
            t1 = std::clamp((-c) / a, 0.0, 1.0);
        } else if (t2 > 1.0) {
            t2 = 1.0;
            t1 = std::clamp((b - c) / a, 0.0, 1.0);
        }
    } 
    // Parallel
    else {
        t1 = std::clamp(-c / a, 0.0, 1.0);
        t2 = std::clamp((b * t1 + f) / e, 0.0, 1.0);
        
        if (t2 == 0.0 || t2 == 1.0) {
             t1 = std::clamp((b * t2 - c) / a, 0.0, 1.0);
        }
    }

    const Eigen::Vector3d closest_p1 = p1 + t1 * d1;
    const Eigen::Vector3d closest_p2 = p2 + t2 * d2;
    double dist = (closest_p1 - closest_p2).norm();

    Eigen::Vector3d normal;
    if (dist > 1e-7) {
        normal = (closest_p1 - closest_p2) / dist;
    } else {
        normal = Eigen::Vector3d::UnitZ();
    }

    return {dist, closest_p1, closest_p2, normal};
}

inline CollisionOutput capsuleDistance(const Capsule &c1, const Capsule &c2) {
    CollisionOutput result = segmentSegmentDistance(
        c1.p1_world, c1.p2_world,
        c2.p1_world, c2.p2_world
    );

    result.distance -= (c1.radius_ + c2.radius_);
    return result;
}

class Robot {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Robot() = default;

    Robot(int robot_id, const std::string &robot_name, const std::string &urdf_path, const Eigen::Vector3d &base_trans,
      const Eigen::Quaterniond &base_quat, const Eigen::VectorXd init_q, bool is_mimic_gripper=true)
    : id_(robot_id), name_(robot_name),
      base_(base_quat.toRotationMatrix() * Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitX()).toRotationMatrix(), base_trans),
      init_q_(init_q),
      is_mimic_gripper_(is_mimic_gripper)
    {
        const std::string urdf_abs = std::filesystem::absolute(urdf_path).string();
        const std::string urdf_dir = std::filesystem::path(urdf_abs).parent_path().string();

        pinocchio::urdf::buildModel(urdf_path, model_);

        data_ = pinocchio::Data(model_);

        std::vector<std::string> mesh_dirs{urdf_dir};
        pinocchio::urdf::buildGeom(model_, urdf_abs, pinocchio::COLLISION, geomModel_, mesh_dirs);

        geomData_ = pinocchio::GeometryData(geomModel_);
        geomModel_.addAllCollisionPairs();
        q_dim_ = model_.nq;
        
        forward_kinematics(init_q_);
        pinocchio::updateGeometryPlacements(model_, data_, geomModel_, geomData_);
        initCapsules();
        current_q = VectorXd::Zero(q_dim_);
        updateRobotJoints(init_q_);
    }

    void forward_kinematics(const Eigen::VectorXd& q) const {
        pinocchio::forwardKinematics(model_, data_, q);
        pinocchio::updateFramePlacements(model_, data_);
    }

    std::vector<Eigen::Vector3d> get_positions(const Eigen::VectorXd& q) const {
        forward_kinematics(q);

        std::vector<Eigen::Vector3d> positions(model_.nframes);
        for (size_t i = 0; i < model_.nframes; ++i) {
            positions[i] = data_.oMf[i].translation();
        }

        return positions;
    }

    std::pair<Eigen::Vector3d, Eigen::Quaterniond> get_eef_positions(const Eigen::VectorXd& q) const {
        forward_kinematics(q);
        const pinocchio::SE3& eef_local = data_.oMf[eef_frame_];
        pinocchio::SE3 eef_world = base_ * eef_local;
        return {eef_world.translation(), Eigen::Quaterniond(eef_world.rotation()).normalized()};
    }

    void updateRobotJoints(const Eigen::VectorXd& q) {
        if (current_q != q) {
            forward_kinematics(q);
            pinocchio::updateFramePlacements(model_, data_);

            for (auto &c : capsules_) {
                c.transform(base_ * data_.oMf[c.parentFrame_]);
            }
            current_q = q;
        }
    }

    void updateRobotViz(const Eigen::VectorXd& q) {
        forward_kinematics(q);
        pinocchio::updateGeometryPlacements(model_, data_, geomModel_, geomData_);
    }

    double selfMinDistance(bool signed_d = true) const {
        double min_d = std::numeric_limits<double>::infinity();
        for (const auto& [i, j] : self_collision_pairs_) {
            double d = capsuleDistance(capsules_[i], capsules_[j]).distance;
            if (d < min_d) min_d = d;
        }

        return min_d;
    }

    const pinocchio::Model& get_model() const {
        return model_;
    }

    pinocchio::Data& get_data() const {
        return data_;
    }

    Dimension get_qDim() const {
        return q_dim_;
    }

    const std::vector<Capsule>& get_capsules() const {
        return capsules_;
    }

    const pinocchio::GeometryModel& get_geom_model() const {
        return geomModel_;
    }
    
    const pinocchio::GeometryData& get_geom_data() const {
        return geomData_;
    }

    const pinocchio::SE3& get_base() const {
        return base_;
    }

    const int get_eef_id() const {
        return eef_frame_;
    }

    const std::vector<size_t>& get_base_id_vec() const {
        return base_id_vec_;
    }

    const std::string& get_name() const {
        return name_;
    }

private:
    // robot info
    int                 id_{-1};
    std::string         name_;
    pinocchio::SE3      base_{pinocchio::SE3::Identity()};
    Dimension           q_dim_{0};
    
    const Eigen::VectorXd   init_q_;
    mutable Eigen::VectorXd current_q;

    pinocchio::Model         model_;
    pinocchio::GeometryModel geomModel_;

    mutable pinocchio::Data  data_;
    pinocchio::GeometryData  geomData_;

    // Capsule collision
    int eef_frame_;
    std::vector<Capsule> capsules_;
    std::set<std::pair<size_t, size_t>> self_collision_pairs_;
    std::vector<size_t> base_id_vec_;

    // mimic gripper
    bool is_mimic_gripper_{true};

    static bool isGripperName(const std::string &n) {
      static const std::regex pat("(finger|knuckle|gripper|robotiq_85)", std::regex::icase);
      return std::regex_search(n, pat);
    }

    static bool isBaseName(const std::string &n) {
      static const std::regex pat("(base)", std::regex::icase);
      return std::regex_search(n, pat);
    }

    void initCapsules() {
        capsules_.clear();
        self_collision_pairs_.clear();
        base_id_vec_.clear();

        bool is_first_gripper = true;
        Eigen::Vector3d gripper_min(std::numeric_limits<double>::max(), 0, 0);
        Eigen::Vector3d gripper_max(std::numeric_limits<double>::lowest(), 0, 0);
        eef_frame_ = -1;

        for (std::size_t gid = 0; gid < geomModel_.geometryObjects.size(); ++gid) {
            const auto& go = geomModel_.geometryObjects[gid];
            go.geometry->computeLocalAABB();
            const auto& aabb = go.geometry->aabb_local;
            
            Eigen::Vector3d extents = aabb.max_ - aabb.min_;
            
            int max_axis;
            extents.maxCoeff(&max_axis);
            
            Eigen::Vector3d p1_local = aabb.center();
            Eigen::Vector3d p2_local = p1_local;

            int axis_a = (max_axis + 1) % 3;
            int axis_b = (max_axis + 2) % 3;
            double radius = std::max(extents[axis_a], extents[axis_b]) * 0.5;
            
            p1_local[max_axis] = aabb.min_[max_axis] + radius;
            p2_local[max_axis] = aabb.max_[max_axis] - radius;

            Eigen::Vector3d p1_local_applied = go.placement.act(p1_local);
            Eigen::Vector3d p2_local_applied = go.placement.act(p2_local);

            if (isBaseName(go.name) && !isGripperName(go.name)) {// ignore base collision with static env
                base_id_vec_.push_back(go.parentFrame);
            }

            if (!isGripperName(go.name) && is_mimic_gripper_) {
                capsules_.emplace_back(p1_local_applied, p2_local_applied, radius, go.name, go.parentFrame);
            }
            else {
                if (is_first_gripper) {
                    eef_frame_ = go.parentFrame;
                    
                    Eigen::Vector3d aabb_min = go.placement.act(aabb.min_);
                    Eigen::Vector3d aabb_max = go.placement.act(aabb.max_);
                    gripper_min = gripper_min.cwiseMin(aabb_min);
                    gripper_max = gripper_max.cwiseMax(aabb_max);

                    is_first_gripper = false;
                }
                else {
                    pinocchio::SE3 wrist_M_joint = model_.frames[eef_frame_].placement.inverse();

                    Eigen::Vector3d aabb_min = go.placement.act(aabb.min_);
                    Eigen::Vector3d aabb_max = go.placement.act(aabb.max_);

                    aabb_min = wrist_M_joint.act(aabb_min);
                    aabb_max = wrist_M_joint.act(aabb_max);

                    gripper_min = gripper_min.cwiseMin(aabb_min);
                    gripper_max = gripper_max.cwiseMax(aabb_max);
                }
            }
            
        }

        // After loop, add the single combined gripper capsule if found
        if (!is_first_gripper) {
            Eigen::Vector3d g_extents = gripper_max - gripper_min;
            int g_max_axis;
            g_extents.maxCoeff(&g_max_axis);
            
            int axis_a = (g_max_axis + 1) % 3;
            int axis_b = (g_max_axis + 2) % 3;
            double g_radius = std::max(g_extents[axis_a], g_extents[axis_b]) * 0.5;

            Eigen::Vector3d g_p1 = (gripper_min + gripper_max) * 0.5;
            Eigen::Vector3d g_p2 = g_p1;
            
            g_p1[g_max_axis] = gripper_min[g_max_axis] + g_radius;
            g_p2[g_max_axis] = gripper_max[g_max_axis] - g_radius;

            capsules_.emplace_back(g_p1, g_p1, g_radius, "combined_gripper", eef_frame_);
        }

        // Exclude Same joint, Parent-Child joints for self collision
        for (size_t i = 0; i < capsules_.size(); ++i) {
            for (size_t j = i + 1; j < capsules_.size(); ++j) {
                const auto& cap1 = capsules_[i];
                const auto& cap2 = capsules_[j];

                pinocchio::JointIndex j1 = model_.frames[cap1.parentFrame_].parent;
                pinocchio::JointIndex j2 = model_.frames[cap2.parentFrame_].parent;

                // Skip if on the same joint
                if (j1 == j2) continue;

                // Skip if they are adjacent joints (Parent-Child)
                if (model_.parents[j1] == j2 || model_.parents[j2] == j1) continue;
                
                // Skip gripper internals (using your existing logic)
                if (isGripperName(cap1.name_) && isGripperName(cap2.name_)) continue;

                self_collision_pairs_.insert({i, j});
            }
        }
    }
};
} // namespace ilqgames
