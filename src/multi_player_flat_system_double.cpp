#include <ilqgames/dynamics/multi_player_flat_system_double.h>
#include <ilqgames/utils/linear_dynamics_approximation_double.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

VectorXd MultiPlayerFlatSystemDouble::Integrate(
        Time time_interval, const VectorXd& xi0,
        const std::vector<VectorXd>& vs) const {
    // Number of integration steps and corresponding time step.
    constexpr size_t kNumIntegrationSteps = 2;
    const double dt = time::kTimeStep / static_cast<Time>(kNumIntegrationSteps);

    CHECK_NOTNULL(continuous_linear_system_.get());
    auto xi_dot = [this, &vs](const VectorXd& xi) {
        VectorXd deriv = this->continuous_linear_system_->A * xi;
        for (size_t ii = 0; ii < NumPlayers(); ii++)
            deriv += this->continuous_linear_system_->Bs[ii] * vs[ii];

        return deriv;
    };    // xi_dot

    // RK4 integration. See https://en.wikipedia.org/wiki/Runge-Kutta_methods for
    // further details.
    VectorXd xi(xi0);
    for (Time t = 0.0; t < time_interval - 0.5 * dt; t += dt) {
        const VectorXd k1 = dt * xi_dot(xi);
        const VectorXd k2 = dt * xi_dot(xi + 0.5 * k1);
        const VectorXd k3 = dt * xi_dot(xi + 0.5 * k2);
        const VectorXd k4 = dt * xi_dot(xi + k3);

        xi += (k1 + 2.0 * (k2 + k3) + k4) / 6.0;
    }

    return xi;
}

}    // namespace ilqgames
