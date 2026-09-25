#ifndef ILQGAMES_DYNAMICS_MULTI_PLAYER_FLAT_SYSTEM_DOUBLE_H
#define ILQGAMES_DYNAMICS_MULTI_PLAYER_FLAT_SYSTEM_DOUBLE_H

#include <ilqgames/dynamics/multi_player_integrable_system_double.h>
#include <ilqgames/utils/linear_dynamics_approximation_double.h>
#include <ilqgames/utils/operating_point_double.h>
#include <ilqgames/utils/quadratic_cost_approximation_double.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <vector>

namespace ilqgames {

class MultiPlayerFlatSystemDouble : public MultiPlayerIntegrableSystemDouble {
 public:
  virtual ~MultiPlayerFlatSystemDouble() {}

  // Compute time derivative of state.
  virtual VectorXd Evaluate(const VectorXd& x,
                            const std::vector<VectorXd>& us) const = 0;

  // Utilities for feedback linearization.
  virtual MatrixXd InverseDecouplingMatrix(const VectorXd& x) const = 0;
  virtual VectorXd AffineTerm(const VectorXd& x) const = 0;
  virtual VectorXd LinearizingControl(const VectorXd& x, const VectorXd& v,
                                      PlayerIndex player) const = 0;
  virtual std::vector<VectorXd> LinearizingControls(
      const VectorXd& x, const std::vector<VectorXd>& vs) const = 0;
  virtual VectorXd ToLinearSystemState(const VectorXd& x) const = 0;
  virtual VectorXd FromLinearSystemState(const VectorXd& xi) const = 0;

  // Gradient and hessian of map from xi to x.
  virtual void ChangeCostCoordinates(
      const VectorXd& xi, std::vector<QuadraticCostApproximationDouble>* q) const = 0;
  virtual void ChangeControlCostCoordinates(
      const VectorXd& xi, std::vector<QuadraticCostApproximationDouble>* q) const = 0;

  // Check if a state is singular.
  virtual bool IsLinearSystemStateSingular(const VectorXd& xi) const = 0;

  // Integrate these dynamics forward in time.
  // Options include integration for a single timestep, between arbitrary times,
  // and within a single timestep.
  VectorXd Integrate(Time time_interval, const VectorXd& xi0,
                     const std::vector<VectorXd>& vs) const;
  VectorXd Integrate(Time t0, Time time_interval, const VectorXd& xi0,
                     const std::vector<VectorXd>& vs) const {
    return Integrate(time_interval, xi0, vs);
  }

  // Can this system be treated as linear for the purposes of LQ solves?
  // For example, linear systems and feedback linearizable systems should return
  // true here.
  bool TreatAsLinear() const { return true; }

  // Getters.
  const LinearDynamicsApproximationDouble& LinearizedSystem() const {
    if (!discrete_linear_system_) ComputeLinearizedSystem();
    return *discrete_linear_system_;
  }

  virtual Dimension UDim(PlayerIndex player_idx) const = 0;
  virtual PlayerIndex NumPlayers() const = 0;

 protected:
  MultiPlayerFlatSystemDouble(Dimension xdim) : MultiPlayerIntegrableSystemDouble(xdim) {}

  // Discrete time approximation of the underlying linearized system.
  virtual void ComputeLinearizedSystem() const = 0;

  // Linearized system (discrete and continuous time).
  mutable std::unique_ptr<const LinearDynamicsApproximationDouble>
      discrete_linear_system_;
  mutable std::unique_ptr<const LinearDynamicsApproximationDouble>
      continuous_linear_system_;

};  //\class MultiPlayerFlatSystemDouble

}  // namespace ilqgames

#endif
