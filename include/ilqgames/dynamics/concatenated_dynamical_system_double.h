#ifndef ILQGAMES_DYNAMICS_CONCATENATED_DYNAMICAL_SYSTEM_DOUBLE_H
#define ILQGAMES_DYNAMICS_CONCATENATED_DYNAMICAL_SYSTEM_DOUBLE_H

#include <ilqgames/dynamics/multi_player_dynamical_system_double.h>
#include <ilqgames/dynamics/single_player_dynamical_system_double.h>
#include <ilqgames/utils/linear_dynamics_approximation_double.h>
#include <ilqgames/utils/types.h>

#include <algorithm>

namespace ilqgames {

using SubsystemList = std::vector<std::shared_ptr<SinglePlayerDynamicalSystemDouble>>;

class ConcatenatedDynamicalSystemDouble : public MultiPlayerDynamicalSystemDouble {
 public:
  ~ConcatenatedDynamicalSystemDouble() {}
  ConcatenatedDynamicalSystemDouble(const SubsystemList& subsystems);

  // Compute time derivative of state.
  VectorXd Evaluate(Time t, const VectorXd& x,
                    const std::vector<VectorXd>& us) const;

  // Compute a discrete-time Jacobian linearization.
  LinearDynamicsApproximationDouble Linearize(Time t, const VectorXd& x,
                                        const std::vector<VectorXd>& us) const;

  // Distance metric between two states.
  double DistanceBetween(const VectorXd& x0, const VectorXd& x1) const;

  // Stitch between two states of the system. Interprets the first one as best
  // for ego and the second as best for other players.
  VectorXd Stitch(const VectorXd& x_ego, const VectorXd& x_others) const {
    VectorXd x(x_ego.size());

    const Dimension ego_state_dim = subsystems_[0]->XDim();
    x.head(ego_state_dim) = x_ego.head(ego_state_dim);
    x.tail(x_others.size() - ego_state_dim) =
        x_others.tail(x_others.size() - ego_state_dim);

    return x;
  }

  // Getters.
  const SubsystemList& Subsystems() const { return subsystems_; }
  PlayerIndex NumPlayers() const { return subsystems_.size(); }
  Dimension SubsystemStartDim(PlayerIndex player_idx) const {
    return subsystem_start_dims_[player_idx];
  }
  Dimension SubsystemXDim(PlayerIndex player_idx) const {
    return subsystems_[player_idx]->XDim();
  }
  Dimension UDim(PlayerIndex player_idx) const {
    return subsystems_[player_idx]->UDim();
  }
  std::vector<Dimension> PositionDimensions() const;

 private:
  // List of subsystems, each of which controls the affects of a single player.
  const SubsystemList subsystems_;

  // Cumulative sum of dimensions of each subsystem.
  std::vector<Dimension> subsystem_start_dims_;
};  // namespace ilqgames

}  // namespace ilqgames

#endif
