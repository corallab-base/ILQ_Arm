#include <ilqgames/dynamics/robot_arm_simple_6d.h>

namespace ilqgames {
// Constexprs for state indices.
const Dimension RobotArmSimple6D::kNumXDims = 12; // state dim = 6 joints + 6 joint vel
const Dimension RobotArmSimple6D::kQ1Idx = 0;
const Dimension RobotArmSimple6D::kQ2Idx = 1;
const Dimension RobotArmSimple6D::kQ3Idx = 2;
const Dimension RobotArmSimple6D::kQ4Idx = 3;
const Dimension RobotArmSimple6D::kQ5Idx = 4;
const Dimension RobotArmSimple6D::kQ6Idx = 5;

const Dimension RobotArmSimple6D::kQd1Idx = 6;
const Dimension RobotArmSimple6D::kQd2Idx = 7;
const Dimension RobotArmSimple6D::kQd3Idx = 8;
const Dimension RobotArmSimple6D::kQd4Idx = 9;
const Dimension RobotArmSimple6D::kQd5Idx = 10;
const Dimension RobotArmSimple6D::kQd6Idx = 11;

// Constexprs for control indices.
const Dimension RobotArmSimple6D::kNumUDims = 6; // control dim = 6 joint acc
const Dimension RobotArmSimple6D::kQdd1Idx = 0;
const Dimension RobotArmSimple6D::kQdd2Idx = 1;
const Dimension RobotArmSimple6D::kQdd3Idx = 2;
const Dimension RobotArmSimple6D::kQdd4Idx = 3;
const Dimension RobotArmSimple6D::kQdd5Idx = 4;
const Dimension RobotArmSimple6D::kQdd6Idx = 5;

}  // namespace ilqgames