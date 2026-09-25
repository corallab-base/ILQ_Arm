#include <ilqgames/cost/quadratic_rob_cost.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <numeric>
#include <cmath>

namespace ilqgames {

double QuadraticRobCost::Evaluate(const VectorXd& input) const {
    CHECK_LE(TIdx_.size(), input.size());

    double j = 0;
    for (size_t i = 0; i < TIdx_.size(); i++) {
        Dimension q = TIdx_[i];

        double delta;
        if (is_relative_)
            delta = std::remainder(input(q) - nominal_(i), 2 * M_PI);
        else
            delta = input(q) - nominal_(i);
        j += 0.5 * weight_(i) * delta * delta;
    }
    return j;
}

void QuadraticRobCost::Quadraticize(const VectorXd& input, MatrixXd* hess,
                                                                 VectorXd* grad) const {
    CHECK_LE(TIdx_.size(), input.size());
    CHECK_NOTNULL(hess);
    CHECK_NOTNULL(grad);

    // Check dimensions.
    CHECK_EQ(input.size(), hess->rows());
    CHECK_EQ(input.size(), hess->cols());
    CHECK_EQ(input.size(), grad->size());

    for (size_t i = 0; i < TIdx_.size(); i++) {
        Dimension q = TIdx_[i];
        double delta;
        if (is_relative_)
            delta = std::remainder(input(q) - nominal_(i), 2 * M_PI);
        else
            delta = input(q) - nominal_(i);
        const double dx = weight_(i) * delta;
        const double ddx = weight_(i);

        (*grad)(q) += dx;
        (*hess)(q, q) += ddx;
    }
}

}    // namespace ilqgames