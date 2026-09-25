#ifdef IMGUI_IMPL_OPENGL_LOADER_GL3W
#undef IMGUI_IMPL_OPENGL_LOADER_GL3W
#endif

#ifdef IMGUI_IMPL_OPENGL_LOADER_GLAD
#undef IMGUI_IMPL_OPENGL_LOADER_GLAD
#endif

#ifndef IMGUI_IMPL_OPENGL_LOADER_GLEW
#define IMGUI_IMPL_OPENGL_LOADER_GLEW
#endif

#include <GL/glew.h>

#include <ilqgames/examples/two_robot_demo4.h>
#include <ilqgames/gui/control_sliders_double.h>
#include <ilqgames/gui/cost_inspector_double.h>
#include <ilqgames/gui/rob_3d_renderer.h>
#include <ilqgames/solver/augmented_lagrangian_solver_double.h>
#include <ilqgames/solver/ilq_solver_double.h>
#include <ilqgames/solver/problem_double.h>
#include <ilqgames/solver/solver_params_double.h>
#include <ilqgames/utils/check_local_nash_equilibrium_double.h>
#include <ilqgames/utils/solver_log_double.h>
#include <ilqgames/utils/signed_distance_field.h>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <memory>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>

DEFINE_bool(open_loop, false, "Use open loop (vs. feedback) solver.");
DEFINE_bool(save, true, "Optionally save solver logs to disk.");
DEFINE_bool(viz, true, "Visualize results in a GUI.");
DEFINE_bool(last_traj, false, "Should the solver only dump the last trajectory?");
DEFINE_string(experiment_name, "two_robot_demo4", "Name for the experiment.");
DEFINE_bool(linesearch, true, "Should the solver linesearch?");
DEFINE_double(initial_alpha_scaling, 0.1, "Initial step size in linesearch.");
DEFINE_double(convergence_tolerance, 1.0, "KKT squared error tolerance.");
DEFINE_double(expected_decrease, 0.001, "KKT sq err expected decrease per iter.");

static void glfw_error_callback(int error, const char* description) {
  fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = true;

  // Setup Problem
  ilqgames::SolverParamsDouble params;
  params.open_loop = false;
  params.linesearch = true;
  params.max_solver_iters = 200;
  params.convergence_tolerance = 1e-1; // increase
  params.max_backtracking_steps = 100;
  params.control_regularization = 1e-3;
  params.initial_alpha_scaling = 1.0;
  params.expected_decrease_fraction = 0.1;
  params.geometric_mu_scaling = 2.0;
  params.geometric_mu_downscaling = 0.2;
  params.geometric_lambda_downscaling = 0.5;
  
  auto problem = std::make_shared<ilqgames::TwoRobotDemo4>();
  problem->Initialize();
  ilqgames::AugmentedLagrangianSolverDouble solver(problem, params);

  // Solve
  auto start = std::chrono::system_clock::now();
  auto log = solver.Solve();
  std::vector<std::shared_ptr<const ilqgames::SolverLogDouble>> logs = {log};
  LOG(INFO) << "Solver finished in " << std::chrono::duration<double>(std::chrono::system_clock::now() - start).count() << "s";

  // Check if solution satisfies sufficient conditions for being a local Nash.
  problem->OverwriteSolution(log->FinalOperatingPoint(),
                             log->FinalStrategies());
  const bool is_local_nash = CheckSufficientLocalNashEquilibrium(*problem);
  if (is_local_nash)
    LOG(INFO) << "Solution is a local Nash.";
  else
    LOG(INFO) << "Solution may not be a local Nash.";

  // Confirm with numerical check.
  constexpr float kMaxPerturbation = 0.1;
  constexpr bool kOpenLoop = false;
  problem->OverwriteSolution(log->FinalOperatingPoint(),
                             log->FinalStrategies());
  const bool is_numerical_nash =
      NumericalCheckLocalNashEquilibriumDouble(*problem, kMaxPerturbation, kOpenLoop);
  if (is_numerical_nash)
    LOG(INFO) << "Solution is a numerical Nash.";
  else
    LOG(INFO) << "Solution is not a numerical Nash.";

  // Dump the logs and/or exit.
  if (FLAGS_save) {
    if (FLAGS_experiment_name == "") {
      CHECK(log->Save(FLAGS_last_traj));
    } else {
      CHECK(log->Save(FLAGS_last_traj, FLAGS_experiment_name));
    }
  }
  if (!FLAGS_viz) return 0;

  // Setup Window
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) return 1;

  const char* glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

  GLFWwindow* window = glfwCreateWindow(1280, 720, "ILQGames 3D", NULL, NULL);
  if (!window) return 1;
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  bool err = false;
  #if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
    err = (gl3wInit() != 0);
  #elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
    err = (glewInit() != GLEW_OK); // This should run now
  #elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
    err = (gladLoadGL() == 0);
  #endif

  if (err) {
      fprintf(stderr, "Failed to initialize OpenGL loader! Check your CMake settings.\n");
      return 1;
  }

  // Setup ImGui
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();

  ImGuiIO& io = ImGui::GetIO();
  // Load a system font at a larger size (e.g., 28 pixels).
  // If this file doesn't exist, check /usr/share/fonts/ for other .ttf files.
  io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20.0f);

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  std::shared_ptr<ilqgames::ControlSlidersDouble> sliders(
      new ilqgames::ControlSlidersDouble({logs}));
  
  std::vector<ilqgames::Robot*> robots = problem->get_robots();
  std::vector<std::shared_ptr<ilqgames::SignedDistanceField>> env = problem->get_sdf();
  ilqgames::Rob3DRenderer rob_3d_render(sliders, {problem}, robots, env);
  ilqgames::CostInspectorDouble cost_inspector(sliders, {problem->PlayerCosts()});

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    sliders->Render();
    cost_inspector.Render();
    rob_3d_render.Render(); 

    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}