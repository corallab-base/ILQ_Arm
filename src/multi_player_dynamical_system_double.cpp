#include <ilqgames/dynamics/multi_player_dynamical_system_double.h>
#include <ilqgames/utils/linear_dynamics_approximation_double.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

VectorXd MultiPlayerDynamicalSystemDouble::Integrate(
        Time t0, Time time_interval, const VectorXd& x0,
        const std::vector<VectorXd>& us) const {
    VectorXd x(x0);

    if (integrate_using_euler_) {
        x += time_interval * Evaluate(t0, x0, us);
    } else {
        // Number of integration steps and corresponding time step.
        constexpr size_t kNumIntegrationSteps = 2;
        const double dt = time_interval / static_cast<Time>(kNumIntegrationSteps);

        // RK4 integration. See https://en.wikipedia.org/wiki/Runge-Kutta_methods
        // for further details.
        for (Time t = t0; t < t0 + time_interval - 0.5 * dt; t += dt) {
            const VectorXd k1 = dt * Evaluate(t, x, us);
            const VectorXd k2 = dt * Evaluate(t + 0.5 * dt, x + 0.5 * k1, us);
            const VectorXd k3 = dt * Evaluate(t + 0.5 * dt, x + 0.5 * k2, us);
            const VectorXd k4 = dt * Evaluate(t + dt, x + k3, us);

            x += (k1 + 2.0 * (k2 + k3) + k4) / 6.0;
        }
    }

    return x;
}

}    // namespace ilqgames
