#ifndef ILQGAMES_DYNAMICS_ROBOT_ARM_SIMPLE_6D_H
#define ILQGAMES_DYNAMICS_ROBOT_ARM_SIMPLE_6D_H

#include <ilqgames/dynamics/single_player_dynamical_system_double.h>
#include <ilqgames/utils/types.h>
#include <ilqgames/utils/robot_arm.h>

#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/aba-derivatives.hpp>

#include <unsupported/Eigen/MatrixFunctions>

using namespace pinocchio;

namespace ilqgames {

class RobotArmSimple6D : public SinglePlayerDynamicalSystemDouble {
 public:
    ~RobotArmSimple6D() {}
    RobotArmSimple6D(int robot_id, const std::string& robot_name, const std::string urdf_path,
                const Vector3d base_trans, const Quaterniond base_quat, const VectorXd init_q)
    : SinglePlayerDynamicalSystemDouble(kNumXDims, kNumUDims), rob_(robot_id, robot_name, urdf_path, base_trans, base_quat, init_q) {
        qDim_ = rob_.get_qDim();
    }

    // Compute time derivative of state.
    VectorXd Evaluate(Time t, const VectorXd& x, const VectorXd& u) const;

    // Compute a discrete-time Jacobian linearization.
    void Linearize(Time t, const VectorXd& x, const VectorXd& u,
                                 Eigen::Ref<MatrixXd> A, Eigen::Ref<MatrixXd> B) const;

    // Distance metric between two states.
    double DistanceBetween(const VectorXd& x0, const VectorXd& x1) const;

    // Position dimensions.
    std::vector<Dimension> PositionDimensions() const { 
        return {
            kQ1Idx,
            kQ2Idx,
            kQ3Idx,
            kQ4Idx,
            kQ5Idx,
            kQ6Idx,
        }; 
    }

    Robot& get_robot() {
        return rob_;
    }


    // Constexprs for state indices.
    static const Dimension kNumXDims; // state dim
    static const Dimension kQ1Idx; // Joint angel
    static const Dimension kQ2Idx; // Joint angel
    static const Dimension kQ3Idx; // Joint angel
    static const Dimension kQ4Idx; // Joint angel
    static const Dimension kQ5Idx; // Joint angel
    static const Dimension kQ6Idx; // Joint angel

    static const Dimension kQd1Idx; // Joint velocity
    static const Dimension kQd2Idx; // Joint velocity
    static const Dimension kQd3Idx; // Joint velocity
    static const Dimension kQd4Idx; // Joint velocity
    static const Dimension kQd5Idx; // Joint velocity
    static const Dimension kQd6Idx; // Joint velocity

    // Constexprs for control indices.
    static const Dimension kNumUDims;
    static const Dimension kQdd1Idx; // Joint acceleration
    static const Dimension kQdd2Idx; // Joint acceleration
    static const Dimension kQdd3Idx; // Joint acceleration
    static const Dimension kQdd4Idx; // Joint acceleration
    static const Dimension kQdd5Idx; // Joint acceleration
    static const Dimension kQdd6Idx; // Joint acceleration
    
    static const std::string urdf_path_;

 private:
    Robot rob_;
    Dimension qDim_;

};    //\class RobotArmSimple6D

inline VectorXd RobotArmSimple6D::Evaluate(Time t, const VectorXd& x, const VectorXd& u) const {
    // x -> 6 * 2 = 12
    // x = [q, qd]
    // u = [qdd] (Simplified Dynamics: Control is acceleration)
    
    Eigen::VectorXd qd = x.tail(qDim_);

    // xdot = [qd, qdd] = [qd, u]
    Eigen::VectorXd xdot(xdim_);
    xdot.head(qDim_) = qd;
    xdot.tail(qDim_) = u;

    return xdot;
}

inline void RobotArmSimple6D::Linearize(Time t, const VectorXd& x, const VectorXd& u,
                                  Eigen::Ref<MatrixXd> A,
                                  Eigen::Ref<MatrixXd> B) const {
    const double dt = time::kTimeStep; 

    // For a simplified double integrator system (q_dd = u):
    // Continuous system:
    // [ q_dot  ]   [ 0  I ] [ q   ]   [ 0 ]
    // [ q_ddot ] = [ 0  0 ] [ q_d ] + [ I ] u
    
    // Discrete system (exact integration):
    // x_{k+1} = A_d * x_k + B_d * u_k
    
    // A_discrete = [ I   I*dt ]
    //              [ 0    I   ]
    
    // B_discrete = [ 0.5*I*dt^2 ]
    //              [    I*dt    ]

    A.setIdentity();
    A.block(0, qDim_, qDim_, qDim_) = Eigen::MatrixXd::Identity(qDim_, qDim_) * dt;

    B.setZero();
    B.block(0, 0, qDim_, qDim_) = Eigen::MatrixXd::Identity(qDim_, qDim_) * (0.5 * dt * dt);
    B.block(qDim_, 0, qDim_, qDim_) = Eigen::MatrixXd::Identity(qDim_, qDim_) * dt;
}

inline double RobotArmSimple6D::DistanceBetween(const VectorXd& x0, const VectorXd& x1) const {
    // Extract joint angles
    Eigen::VectorXd q0 = x0.head(qDim_);
    Eigen::VectorXd q1 = x1.head(qDim_);

    std::vector<Eigen::Vector3d> positions0 = rob_.get_positions(q0);
    std::vector<Eigen::Vector3d> positions1 = rob_.get_positions(q1);

    // Sum squared distances over all frames
    double dist_sq = 0.0;
    for (size_t i = 0; i < positions0.size(); ++i) {
        Eigen::Vector3d dp = positions0[i] - positions1[i];
        dist_sq += dp.squaredNorm();
    }

    return dist_sq;
}

}    // namespace ilqgames

#endif
