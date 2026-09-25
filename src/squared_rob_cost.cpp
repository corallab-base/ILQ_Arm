#include <ilqgames/cost/squared_rob_cost.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <numeric>

namespace ilqgames {

double SquaredRobCost::Evaluate(const VectorXd& input) const {
    CHECK_EQ(TIdx_.size(), input.size());

    double j = 0;
    for (auto i : TIdx_) {
        j += weight_(i) * (input(i) * input(i));
    }

    return j;
}

void SquaredRobCost::Quadraticize(const VectorXd& input, MatrixXd* hess,
                                 VectorXd* grad) const {
    CHECK_EQ(TIdx_.size(), input.size());
    CHECK_NOTNULL(hess);
    CHECK_NOTNULL(grad);

    // Check dimensions.
    CHECK_EQ(input.size(), hess->rows());
    CHECK_EQ(input.size(), hess->cols());
    CHECK_EQ(input.size(), grad->size());

    for (auto i : TIdx_) {
        const double x = input(i);
        const double w = weight_(i);

        (*grad)(i) += 2.0 * w * x;
        (*hess)(i, i) += 2.0 * w;
    }
}

}  // namespace ilqgames
