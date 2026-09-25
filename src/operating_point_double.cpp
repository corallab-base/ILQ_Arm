#include <ilqgames/dynamics/multi_player_dynamical_system_double.h>
#include <ilqgames/utils/operating_point_double.h>
#include <ilqgames/utils/types.h>

#include <vector>

namespace ilqgames {

OperatingPointDouble::OperatingPointDouble(size_t num_time_steps, PlayerIndex num_players, Time initial_time)
    : xs(num_time_steps), us(num_time_steps), t0(initial_time) {
  for (auto& entry : us) entry.resize(num_players);
}

void OperatingPointDouble::swap(OperatingPointDouble& other) {
  xs.swap(other.xs);
  us.swap(other.us);
  std::swap(t0, other.t0);
}

}  // namespace ilqgames
