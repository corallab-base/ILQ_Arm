#include <ilqgames/dynamics/concatenated_dynamical_system_double.h>
#include <ilqgames/utils/linear_dynamics_approximation_double.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

ConcatenatedDynamicalSystemDouble::ConcatenatedDynamicalSystemDouble(
    const SubsystemList& subsystems)
    : MultiPlayerDynamicalSystemDouble(std::accumulate(
          subsystems.begin(), subsystems.end(), 0,
          [](Dimension total,
             const std::shared_ptr<SinglePlayerDynamicalSystemDouble>& subsystem) {
            CHECK_NOTNULL(subsystem.get());
            return total + subsystem->XDim();
          })),
      subsystems_(subsystems) {
  // Populate subsystem start dimensions.
  subsystem_start_dims_.push_back(0);
  for (const auto& subsystem : subsystems_) {
    subsystem_start_dims_.push_back(subsystem_start_dims_.back() +
                                    subsystem->XDim());
  }
}

VectorXd ConcatenatedDynamicalSystemDouble::Evaluate(
    Time t, const VectorXd& x, const std::vector<VectorXd>& us) const {
  CHECK_EQ(us.size(), NumPlayers());

  // Populate 'xdot' one subsystem at a time.
  VectorXd xdot(xdim_);
  Dimension dims_so_far = 0;
  for (size_t ii = 0; ii < NumPlayers(); ii++) {
    const auto& subsystem = subsystems_[ii];
    xdot.segment(dims_so_far, subsystem->XDim()) = subsystem->Evaluate(
        t, x.segment(dims_so_far, subsystem->XDim()), us[ii]);
    dims_so_far += subsystem->XDim();
  }

  return xdot;
}

LinearDynamicsApproximationDouble ConcatenatedDynamicalSystemDouble::Linearize(
    Time t, const VectorXd& x, const std::vector<VectorXd>& us) const {
  CHECK_EQ(us.size(), NumPlayers());

  // Populate a block-diagonal A, as well as Bs.
  LinearDynamicsApproximationDouble linearization(*this);

  Dimension dims_so_far = 0;
  for (size_t ii = 0; ii < NumPlayers(); ii++) {
    const auto& subsystem = subsystems_[ii];
    const Dimension xdim = subsystem->XDim();
    const Dimension udim = subsystem->UDim();
    subsystem->Linearize(
        t, x.segment(dims_so_far, xdim), us[ii],
        linearization.A.block(dims_so_far, dims_so_far, xdim, xdim),
        linearization.Bs[ii].block(dims_so_far, 0, xdim, udim));

    dims_so_far += subsystem->XDim();
  }

  return linearization;
}

double ConcatenatedDynamicalSystemDouble::DistanceBetween(const VectorXd& x0,
                                                   const VectorXd& x1) const {
  Dimension dims_so_far = 0;
  float total = 0.0;

  // Accumulate total across all subsystems.
  for (const auto& subsystem : subsystems_) {
    const Dimension xdim = subsystem->XDim();
    total += subsystem->DistanceBetween(x0.segment(dims_so_far, xdim),
                                        x1.segment(dims_so_far, xdim));

    dims_so_far += xdim;
  }

  return total;
}

std::vector<Dimension> ConcatenatedDynamicalSystemDouble::PositionDimensions() const {
  std::vector<Dimension> dims;

  for (const auto& s : subsystems_) {
    const std::vector<Dimension> sub_dims = s->PositionDimensions();
    dims.insert(dims.end(), sub_dims.begin(), sub_dims.end());
  }

  return dims;
}

}  // namespace ilqgames
