#ifndef ILQGAMES_UTILS_STRATEGY_DOUBLE_H
#define ILQGAMES_UTILS_STRATEGY_DOUBLE_H

#include <ilqgames/utils/operating_point_double.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <vector>

namespace ilqgames {

struct StrategyDouble {
  std::vector<MatrixXd> Ps;
  std::vector<VectorXd> alphas;

  // Preallocate memory during construction.
  StrategyDouble(size_t horizon, Dimension xdim, Dimension udim)
      : Ps(horizon), alphas(horizon) {
    for (size_t ii = 0; ii < horizon; ii++) {
      Ps[ii] = MatrixXd::Zero(udim, xdim);
      alphas[ii] = VectorXd::Zero(udim);
    }
  }

  // Operator for computing control given time index and delta x.
  VectorXd operator()(size_t time_index, const VectorXd& delta_x,
                      const VectorXd& u_ref) const {
    return u_ref - Ps[time_index] * delta_x - alphas[time_index];
  }

  // Number of variables.
  size_t NumVariables() const {
    const size_t horizon = Ps.size();
    CHECK_EQ(horizon, alphas.size());

    return horizon * (Ps.front().size() + alphas.front().size());
  }
};  // struct StrategyDouble

}  // namespace ilqgames

#endif
