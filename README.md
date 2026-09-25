# Coordinated Motion Planning for Multi-Arm Systems via Iterative LQ Games

This repository contains the code for the paper
[**Coordinated Motion Planning for Multi-Arm Systems via Iterative LQ Games**](https://arxiv.org/abs/2608.27726)
by Junyoung Kim, Hanwen Ren, Lei Zhang, and Ahmed H. Qureshi.

Each manipulator is an independent agent with its own objective. The planner solves a
sequence of local linear-quadratic (LQ) games around the current trajectory, and Riccati
recursions give feedback Nash strategies for all arms. Differentiable collision penalties
cover self-collision, arm-to-arm collision, and collision with the environment.

The code builds on the [ilqgames](https://github.com/HJReachability/ilqgames) library by
David Fridovich-Keil et al. It adds 6-DoF robot arm dynamics (UR5e), collision handling
through Pinocchio and Coal, and a 3D visualizer.

## Dependencies

Tested on Ubuntu with CMake ≥ 3.22 and a C++17 compiler.

- `eigen3`, `glog`, `gflags`
- `opengl`, `glut`, `glew`
- [Coal](https://github.com/coal-library/coal) (collision checking)
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio) (robot kinematics), built with collision support

GLFW, Dear ImGui, and GoogleTest come bundled in `external/`.

```bash
sudo apt install libeigen3-dev libgoogle-glog-dev libgflags-dev \
                 freeglut3-dev libglew-dev xorg-dev libglu1-mesa-dev fonts-dejavu-core
```

Build Coal and Pinocchio from source and install them to `/usr/local`:

```bash
# Coal
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
         -DCOAL_BACKWARD_COMPATIBILITY_WITH_HPP_FCL=ON -DBUILD_PYTHON_INTERFACE=OFF
# Pinocchio
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
         -DBUILD_WITH_COLLISION_SUPPORT=ON -DBUILD_PYTHON_INTERFACE=OFF
```

## Build

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/usr/local -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

Executables are written to `bin/`. The unit tests are built as `build/run_tests`.

## Running

Each folder in `exec/` becomes one executable. For example:

```bash
./bin/two_robot_demo1
```

The program solves the game, reports whether the solution is a local Nash equilibrium,
saves the solver log under `logs/`, and opens a 3D viewer where you can step through the
solver iterations. Common flags:

```bash
./bin/two_robot_demo1 --viz=false              # solve only, no GUI
./bin/two_robot_demo1 --save=false             # don't write logs
./bin/two_robot_demo1 --experiment_name=my_run # set the log folder name
./bin/<executable> --help                      # list all flags
```

### Included scenarios

| Executable | Description |
|---|---|
| `two_robot_demo{1-4}`, `three_robot_demo{1,2}` | Hand-designed 2- and 3-arm scenarios |
| `two_robot_complex_env_demo{1-4}`, `three_robot_complex_env` | Scenarios with obstacles in the workspace |
| `two_robot_cross_{easy,hard}`, `two_robot_intersect_easy` | Arms whose paths cross each other |
| `multi_robot_random_eval`, `multi_robot_random_complex_env` | Batch evaluation on random start/goal sets |

### Batch evaluation

The random start/goal test sets used in the paper are in `random_tests/test_cases/`. Each
evaluation executable loads one set (select it by editing the test parameters near the top
of its `main.cpp`), runs every case, and writes per-case logs and a summary to
`random_tests/test_results/`.

To generate a new test set, use `bin/gen_random_start_goal` or
`bin/gen_complex_env_start_goal`. Their configuration is set in
`test/gen_random_start_goal.cpp` and `test/gen_complex_env_start_goal.cpp`.

## Repository layout

```
include/ilqgames/, src/   Solver, dynamics, costs, constraints, GUI, and example problems
exec/                     One main.cpp per executable
test/                     Unit tests and test-set generators
robot/ur5e/               UR5e URDF and meshes
random_tests/test_cases/  Evaluation test sets
external/                 Bundled third-party libraries
```

## Citation

```bibtex
@article{kim2026coordinated,
  title   = {Coordinated Motion Planning for Multi-Arm Systems via Iterative LQ Games},
  author  = {Kim, Junyoung and Ren, Hanwen and Zhang, Lei and Qureshi, Ahmed H.},
  journal = {arXiv preprint arXiv:2608.27726},
  year    = {2026}
}
```

This work builds on iterative LQ games. Please also cite:

```bibtex
@inproceedings{fridovich2020efficient,
  title     = {Efficient iterative linear-quadratic approximations for nonlinear multi-player general-sum differential games},
  author    = {Fridovich-Keil, David and Ratner, Ellis and Peters, Lasse and Dragan, Anca D and Tomlin, Claire J},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  pages     = {1475--1481},
  year      = {2020}
}
```

## License

Parts of this code are derived from [ilqgames](https://github.com/HJReachability/ilqgames)
and remain under the BSD 3-Clause License of The Regents of the University of California.
See [LICENSE-ilqgames](LICENSE-ilqgames). The libraries bundled in `external/` include
their own license files.
