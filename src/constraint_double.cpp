#include <ilqgames/constraint/constraint_double.h>
#include <ilqgames/utils/relative_time_tracker.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

double ConstraintDouble::mu_ = constants::kDefaultMu;

void ConstraintDouble::ModifyDerivatives(Time t, double g, double* dx, double* ddx,
                                   double* dy, double* ddy, double* dxdy) const {
  // Unpack lambda.
  const double lambda = lambdas_[TimeIndex(t)];
  const double mu = Mu(lambda, g);

  // Assumes that these are just the derivatives of g(x, y), and modifies them
  // to be derivatives of lambda g(x) + mu g(x) g(x) / 2.
  const double new_dx = lambda * *dx + mu * g * *dx;
  const double new_ddx = lambda * *ddx + mu * (*dx * *dx + g * *ddx);

  if (dy) {
    CHECK_NOTNULL(ddy);
    CHECK_NOTNULL(dxdy);

    const double new_dy = lambda * *dy + mu * g * *dy;
    const double new_ddy = lambda * *ddy + mu * (*dy * *dy + g * *ddy);
    const double new_dxdy = lambda * *dxdy + mu * (*dy * *dx + g * *dxdy);

    *dy = new_dy;
    *ddy = new_ddy;
    *dxdy = new_dxdy;
  }

  *dx = new_dx;
  *ddx = new_ddx;
}

}  // namespace ilqgames
