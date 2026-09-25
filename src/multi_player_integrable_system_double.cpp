#include <ilqgames/dynamics/multi_player_integrable_system_double.h>
#include <ilqgames/utils/operating_point_double.h>
#include <ilqgames/utils/strategy_double.h>
#include <ilqgames/utils/types.h>

#include <vector>

namespace ilqgames {

bool MultiPlayerIntegrableSystemDouble::integrate_using_euler_ = false;

VectorXd MultiPlayerIntegrableSystemDouble::Integrate(
        Time t0, Time t, const VectorXd& x0, const OperatingPointDouble& operating_point,
        const std::vector<StrategyDouble>& strategies) const {
    CHECK_GE(t, t0);
    CHECK_GE(t0, operating_point.t0);
    CHECK_EQ(strategies.size(), NumPlayers());

    std::vector<VectorXd> us(NumPlayers());

    // Compute current timestep and final timestep.
    const Time relative_t0 = t0 - operating_point.t0;
    const size_t current_timestep =
            static_cast<size_t>(relative_t0 / time::kTimeStep);

    const Time relative_t = t - operating_point.t0;
    const size_t final_timestep =
            static_cast<size_t>(relative_t / time::kTimeStep);

    // Handle case where 't0' is after 'operating_point.t0' by integrating from
    // 't0' to the next discrete timestep.
    VectorXd x(x0);
    if (t0 > operating_point.t0)
        x = IntegrateToNextTimeStep(t0, x0, operating_point, strategies);

    // Integrate forward step by step up to timestep including t.
    x = Integrate(current_timestep + 1, final_timestep, x, operating_point,
                                strategies);

    // Integrate forward from this timestep to t.
    return IntegrateFromPriorTimeStep(t, x, operating_point, strategies);
}

VectorXd MultiPlayerIntegrableSystemDouble::Integrate(
        size_t initial_timestep, size_t final_timestep, const VectorXd& x0,
        const OperatingPointDouble& operating_point,
        const std::vector<StrategyDouble>& strategies) const {
    VectorXd x(x0);
    std::vector<VectorXd> us(NumPlayers());
    for (size_t kk = initial_timestep; kk < final_timestep; kk++) {
        const Time t = operating_point.t0 + kk * time::kTimeStep;

        // Populate controls for all players.
        for (PlayerIndex ii = 0; ii < NumPlayers(); ii++)
            us[ii] = strategies[ii](kk, x - operating_point.xs[kk],
                                                            operating_point.us[kk][ii]);

        x = Integrate(t, time::kTimeStep, x, us);
    }

    return x;
}

VectorXd MultiPlayerIntegrableSystemDouble::IntegrateToNextTimeStep(
        Time t0, const VectorXd& x0, const OperatingPointDouble& operating_point,
        const std::vector<StrategyDouble>& strategies) const {
    CHECK_GE(t0, operating_point.t0);

    // Compute remaining time this timestep.
    const Time relative_t0 = t0 - operating_point.t0;
    const size_t current_timestep = static_cast<size_t>(
            (relative_t0 +
             constants::kSmallNumber)    // Add to avoid inadvertently subtracting 1.
            / time::kTimeStep);
    const Time remaining_time_this_step =
            time::kTimeStep * (current_timestep + 1) - relative_t0;
    CHECK_LT(remaining_time_this_step, time::kTimeStep + constants::kSmallNumber);
    CHECK_LT(current_timestep, operating_point.xs.size());

    // Interpolate x0_ref.
    const double frac = remaining_time_this_step / time::kTimeStep;
    const VectorXd x0_ref =
            (current_timestep + 1 < operating_point.xs.size())
                    ? frac * operating_point.xs[current_timestep] +
                                (1.0 - frac) * operating_point.xs[current_timestep + 1]
                    : operating_point.xs.back();

    // Populate controls for each player.
    std::vector<VectorXd> us(NumPlayers());
    for (PlayerIndex ii = 0; ii < NumPlayers(); ii++)
        us[ii] = strategies[ii](current_timestep, x0 - x0_ref,
                                                        operating_point.us[current_timestep][ii]);

    return Integrate(t0, remaining_time_this_step, x0, us);
}

VectorXd MultiPlayerIntegrableSystemDouble::IntegrateFromPriorTimeStep(
        Time t, const VectorXd& x0, const OperatingPointDouble& operating_point,
        const std::vector<StrategyDouble>& strategies) const {
    // Compute time until next timestep.
    const Time relative_t = t - operating_point.t0;
    const size_t current_timestep =
            static_cast<size_t>(relative_t / time::kTimeStep);
    const Time remaining_time_until_t =
            relative_t - time::kTimeStep * current_timestep;
    CHECK_LT(current_timestep, operating_point.xs.size()) << t;
    CHECK_LT(remaining_time_until_t, time::kTimeStep);

    // Populate controls for each player.
    std::vector<VectorXd> us(NumPlayers());
    for (PlayerIndex ii = 0; ii < NumPlayers(); ii++) {
        us[ii] = strategies[ii](current_timestep,
                                                        x0 - operating_point.xs[current_timestep],
                                                        operating_point.us[current_timestep][ii]);
    }

    return Integrate(operating_point.t0 + time::kTimeStep * current_timestep,
                                     remaining_time_until_t, x0, us);
}

VectorXd MultiPlayerIntegrableSystemDouble::Integrate(
        Time t0, Time time_interval, const Eigen::Ref<VectorXd>& x0,
        const std::vector<Eigen::Ref<VectorXd>>& us) const {
    std::vector<VectorXd> eval_us(us.size());
    std::transform(us.begin(), us.end(), eval_us.begin(),
                                 [](const Eigen::Ref<VectorXd>& u) { return u.eval(); });

    return Integrate(t0, time_interval, x0.eval(), eval_us);
};

}    // namespace ilqgames
