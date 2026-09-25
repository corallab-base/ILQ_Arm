#ifndef ILQGAMES_DYNAMICS_MULTI_PLAYER_INTEGRABLE_SYSTEM_DOUBLE_H
#define ILQGAMES_DYNAMICS_MULTI_PLAYER_INTEGRABLE_SYSTEM_DOUBLE_H

#include <ilqgames/utils/operating_point_double.h>
#include <ilqgames/utils/strategy_double.h>
#include <ilqgames/utils/types.h>

#include <vector>

namespace ilqgames {

class MultiPlayerIntegrableSystemDouble {
 public:
  virtual ~MultiPlayerIntegrableSystemDouble() {}

  // Integrate these dynamics forward in time.
  // Options include integration for a single timestep, between arbitrary times,
  // and within a single timestep.
  virtual VectorXd Integrate(Time t0, Time time_interval, const VectorXd& x0,
                             const std::vector<VectorXd>& us) const = 0;
  VectorXd Integrate(Time t0, Time t, const VectorXd& x0,
                     const OperatingPointDouble& operating_point,
                     const std::vector<StrategyDouble>& strategies) const;
  VectorXd Integrate(size_t initial_timestep, size_t final_timestep,
                     const VectorXd& x0, const OperatingPointDouble& operating_point,
                     const std::vector<StrategyDouble>& strategies) const;
  VectorXd IntegrateToNextTimeStep(
      Time t0, const VectorXd& x0, const OperatingPointDouble& operating_point,
      const std::vector<StrategyDouble>& strategies) const;
  VectorXd IntegrateFromPriorTimeStep(
      Time t, const VectorXd& x0, const OperatingPointDouble& operating_point,
      const std::vector<StrategyDouble>& strategies) const;

  // Make a utility version of the above that operates on Eigen::Refs.
  VectorXd Integrate(Time t0, Time time_interval,
                     const Eigen::Ref<VectorXd>& x0,
                     const std::vector<Eigen::Ref<VectorXd>>& us) const;

  // Can this system be treated as linear for the purposes of LQ solves?
  // For example, linear systems and feedback linearizable systems should
  // return true here.
  virtual bool TreatAsLinear() const { return false; }

  // Stitch between two states of the system. By default, just takes the
  // first one but concatenated systems, e.g., can interpret the first one
  // as best for ego and the second as best for other players.
  virtual VectorXd Stitch(const VectorXd& x_ego,
                          const VectorXd& x_others) const {
    return x_ego;
  }

  // Integrate using single step Euler or not, see below for more extensive
  // description.
  static void IntegrateUsingEuler() { integrate_using_euler_ = true; }
  static void IntegrateUsingRK4() { integrate_using_euler_ = false; }
  static bool IntegrationUsesEuler() { return integrate_using_euler_; }

  // Getters.
  Dimension XDim() const { return xdim_; }
  Dimension TotalUDim() const {
    Dimension total = 0;
    for (PlayerIndex ii = 0; ii < NumPlayers(); ii++) total += UDim(ii);
    return total;
  }
  virtual Dimension UDim(PlayerIndex player_idx) const = 0;
  virtual PlayerIndex NumPlayers() const = 0;
  virtual std::vector<Dimension> PositionDimensions() const = 0;

  // Distance metric between two states. By default, just the *squared* 2-norm.
  virtual double DistanceBetween(const VectorXd& x0, const VectorXd& x1) const {
    return (x0 - x1).squaredNorm();
  }

 protected:
  MultiPlayerIntegrableSystemDouble(Dimension xdim) : xdim_(xdim) {}

  // State dimension.
  const Dimension xdim_;

  // Whether to use single Euler during integration. Typically this is false but
  // it is typically used either for testing (we only derive Nash typically in
  // this case) or for speed.
  static bool integrate_using_euler_;
};  //\class MultiPlayerIntegrableSystemDouble

}  // namespace ilqgames

#endif
