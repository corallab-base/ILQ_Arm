#ifndef ILQGAMES_UTILS_OPERATING_POINT_DOUBLE_H
#define ILQGAMES_UTILS_OPERATING_POINT_DOUBLE_H

#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <vector>

namespace ilqgames {

struct OperatingPointDouble {
  // Time-indexed list of states.
  std::vector<VectorXd> xs;

  // Time-indexed list of controls for all players, i.e. us[kk] is the list of
  // controls for all players at time index kk.
  std::vector<std::vector<VectorXd>> us;

  // Initial time stamp.
  Time t0;

  // Construct with empty vectors of the right size, and optionally zero out if
  // dynamics is non-null.
  OperatingPointDouble(size_t num_time_steps, PlayerIndex num_players,
                 Time initial_time);

  template <typename MultiPlayerSystemType>
  OperatingPointDouble(size_t num_time_steps, Time initial_time,
                 const std::shared_ptr<const MultiPlayerSystemType>& dynamics)
      : OperatingPointDouble(num_time_steps, dynamics->NumPlayers(), initial_time) {
    CHECK_NOTNULL(dynamics.get());
    for (size_t kk = 0; kk < num_time_steps; kk++) {
      xs[kk] = VectorXd::Zero(dynamics->XDim());
      for (PlayerIndex ii = 0; ii < dynamics->NumPlayers(); ii++)
        us[kk][ii] = VectorXd::Zero(dynamics->UDim(ii));
    }
  }

  // Custom swap function.
  void swap(OperatingPointDouble& other);
};  // struct OperatingPointDouble

}  // namespace ilqgames

#endif
