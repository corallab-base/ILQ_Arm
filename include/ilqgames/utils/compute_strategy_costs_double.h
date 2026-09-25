/*
 * Copyright (c) 2020, The Regents of the University of California (Regents).
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Please contact the author(s) of this library if you have any questions.
 * Authors: David Fridovich-Keil   ( dfk@eecs.berkeley.edu )
 */

///////////////////////////////////////////////////////////////////////////////
//
// Compute costs for each player associated with this set of strategies and
// operating point.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_UTILS_COMPUTE_STRATEGY_COSTS_DOUBLE_H
#define ILQGAMES_UTILS_COMPUTE_STRATEGY_COSTS_DOUBLE_H

#include <ilqgames/cost/player_cost_double.h>
#include <ilqgames/dynamics/multi_player_flat_system_double.h>
#include <ilqgames/dynamics/multi_player_integrable_system_double.h>
#include <ilqgames/solver/problem_double.h>
#include <ilqgames/utils/operating_point_double.h>
#include <ilqgames/utils/quadratic_cost_approximation_double.h>
#include <ilqgames/utils/strategy_double.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <Eigen/Dense>
#include <random>
#include <vector>

namespace ilqgames {

// Compute cost of a set of strategies for each player.
std::vector<double> ComputeStrategyCostsDouble(
    const std::vector<PlayerCostDouble>& player_costs,
    const std::vector<StrategyDouble>& strategies,
    const OperatingPointDouble& operating_point,
    const MultiPlayerIntegrableSystemDouble& dynamics, const VectorXd& x0,
    bool open_loop = false);
std::vector<double> ComputeStrategyCostsDouble(const ProblemDouble& problem,
                                        bool open_loop = false);

}  // namespace ilqgames

#endif
