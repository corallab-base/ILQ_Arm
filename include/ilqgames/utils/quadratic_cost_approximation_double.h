#ifndef ILQGAMES_UTILS_QUADRATIC_COST_APPROXIMATION_DOUBLE_H
#define ILQGAMES_UTILS_QUADRATIC_COST_APPROXIMATION_DOUBLE_H

#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <unordered_map>

namespace ilqgames {

struct SingleCostApproximationDouble {
  MatrixXd hess;
  VectorXd grad;

  // Construct from matrix/vector directly.
  SingleCostApproximationDouble(const MatrixXd& hessian, const VectorXd& gradient)
      : hess(hessian), grad(gradient) {
    CHECK_EQ(hess.rows(), hess.cols());
    CHECK_EQ(hess.rows(), grad.size());
  }

  // Construct with zeros.
  SingleCostApproximationDouble(Dimension dim, double regularization = 0.0)
      : hess(regularization * MatrixXd::Identity(dim, dim)),
        grad(VectorXd::Zero(dim)) {}
};  // struct SingleCostApproximationDouble

struct QuadraticCostApproximationDouble {
  SingleCostApproximationDouble state;
  PlayerMap<SingleCostApproximationDouble> control;

  // Construct from state dimension.
  explicit QuadraticCostApproximationDouble(Dimension xdim,
                                      double regularization = 0.0)
      : state(xdim, regularization) {}
};  // struct QuadraticCostApproximationDouble

}  // namespace ilqgames

#endif
