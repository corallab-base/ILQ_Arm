#ifndef ILQGAMES_UTILS_LINEAR_DYNAMICS_APPROXIMATION_DOUBLE_H
#define ILQGAMES_UTILS_LINEAR_DYNAMICS_APPROXIMATION_DOUBLE_H

#include <ilqgames/utils/types.h>
#include <vector>

namespace ilqgames {

struct LinearDynamicsApproximationDouble {
  MatrixXd A;
  std::vector<MatrixXd> Bs;

  // Default constructor.
  LinearDynamicsApproximationDouble() {}

  // Construct from a MultiPlayerDynamicalSystem. Templated to avoid include
  // cycle. Initialize A to identity and Bs to zero (since this is for a
  // discrete-time linearization).
  template <typename MultiPlayerSystemType>
  explicit LinearDynamicsApproximationDouble(const MultiPlayerSystemType& system)
      : A(MatrixXd::Identity(system.XDim(), system.XDim())),
        Bs(system.NumPlayers()) {
    for (size_t ii = 0; ii < system.NumPlayers(); ii++)
      Bs[ii] = MatrixXd::Zero(system.XDim(), system.UDim(ii));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};  // struct LinearDynamicsApproximationDouble

}  // namespace ilqgames

#endif
