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

#include <ilqgames/examples/multi_robot_eval.h>
#include <ilqgames/gui/control_sliders_double.h>
#include <ilqgames/gui/cost_inspector_double.h>
#include <ilqgames/gui/rob_3d_renderer.h>
#include <ilqgames/solver/augmented_lagrangian_solver_double.h>
#include <ilqgames/solver/ilq_solver_double.h>
#include <ilqgames/solver/problem_double.h>
#include <ilqgames/solver/solver_params_double.h>
#include <ilqgames/utils/check_local_nash_equilibrium_double.h>
#include <ilqgames/utils/solver_log_double.h>
#include <ilqgames/utils/make_directory.h>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <ilqgames/utils/binary_save.h>
#include <ilqgames/utils/test_eval_recorder.h>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>

DEFINE_bool(open_loop, false, "Use open loop (vs. feedback) solver.");

DEFINE_bool(save, true, "Optionally save solver logs to disk.");
DEFINE_bool(viz, true, "Visualize results in a GUI.");
DEFINE_bool(last_traj, false, "Should the solver only dump the last trajectory?");

DEFINE_string(experiment_name, "multi_robot", "Name for the experiment.");

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

    // test params
    size_t num_cases = 250;
    size_t num_robots = 2;
    size_t rob_dim = 6;
    bool is_complex_env = true;
    bool is_floor_on = true;
    bool is_box_on = true;
    bool is_move_on = false;

    // read testcases
    std::string test_case_name = std::to_string(num_cases) + "_" + std::to_string(num_robots) + "rob_complex_env_test";
    // std::string test_case_name = std::to_string(num_cases) + "_" + std::to_string(num_robots) + "rob_complex_ablation";


    std::string test_case_dir = std::string(ILQGAMES_TEST_DIR) + "/test_cases/" + test_case_name;
    std::cout << "Reading: " << test_case_dir << std::endl;
    std::vector<Eigen::VectorXd> test_cases = ilqgames::loadBinary(test_case_dir + "/start_goal.bin");
    if (test_cases.empty()) {
        LOG(ERROR) << "No test cases loaded!";
        return 1;
    }


    // read configs
    std::vector<Eigen::VectorXd> rob_configs = ilqgames::loadBinary(test_case_dir + "/rob_configs.bin");
    if (rob_configs.empty()) {
        LOG(ERROR) << "No rob config loaded!";
        return 1;
    }
    std::vector<Eigen::VectorXd> env_configs = {Eigen::VectorXd::Zero(8)};
    if (is_box_on || is_move_on) {
        env_configs = ilqgames::loadBinary(test_case_dir + "/env_configs.bin");
        if (env_configs.empty()) {
            LOG(ERROR) << "No env config loaded!";
            return 1;
        }
    }

    std::vector<Eigen::Vector3d> base_trans;
    std::vector<Eigen::Quaterniond> base_quats;

    
    for (const auto& config : rob_configs) {
        // config is size 7: [x, y, z, w, x, y, z]
        base_trans.emplace_back(config(0), config(1), config(2));
        base_quats.emplace_back(config(3), config(4), config(5), config(6));
    }

    // make test dir
    ilqgames::MakeDirectory(std::string(ILQGAMES_TEST_DIR));
    const auto now = std::chrono::system_clock::now();
    const std::time_t t_c = std::chrono::system_clock::to_time_t(now);
    std::tm ltm = *std::localtime(&t_c);
    std::ostringstream oss;
    oss << std::put_time(&ltm, "%m-%d-%H:%M");
    std::string dateTimeString = oss.str();

    const std::string Test_dir = FLAGS_experiment_name + "_" + test_case_name;
    ilqgames::MakeDirectory(std::string(ILQGAMES_TEST_DIR) + "/test_results/");
    ilqgames::MakeDirectory(std::string(ILQGAMES_TEST_DIR) + "/test_results/" + Test_dir);

    const std::string Test_name = "test_results/" + FLAGS_experiment_name + "_" + test_case_name + "/" + dateTimeString;
    ilqgames::MakeDirectory(std::string(ILQGAMES_TEST_DIR) + "/" + Test_name);

    ilqgames::TestEvalRecorder recorder(test_cases.size()/2, params.max_solver_iters);
    std::vector<std::vector<std::shared_ptr<const ilqgames::SolverLogDouble>>> all_problems_logs;
    std::vector<std::shared_ptr<const ilqgames::TopDownRobRenderableProblem>> all_problems;
    std::vector<ilqgames::Robot*> robots_for_viz;
    std::vector<std::shared_ptr<ilqgames::SignedDistanceField>> env;
    std::vector<ilqgames::MovingObs> moving_obs;
    size_t case_id = 1;
    for (size_t i = 0; i < test_cases.size(); i+=2) {
        const Eigen::VectorXd& start_q = test_cases[i];
        const Eigen::VectorXd& goal_q = test_cases[i+1];

        // Validate Input Size (Expect 6*4 = 24 dims)
        if (start_q.size() != rob_dim * num_robots) {
            LOG(WARNING) << "Skipping index " << i << ": Expected size" << rob_dim * num_robots << ", got " << start_q.size();
            continue;
        }
        if (goal_q.size() != rob_dim * num_robots) {
            LOG(WARNING) << "Skipping index " << i << ": Expected size" << rob_dim * num_robots << ", got " << goal_q.size();
            continue;
        }


        // Parse Vectors
        size_t spit = 0;
        std::vector<Eigen::VectorXd>starts(num_robots);
        std::vector<Eigen::VectorXd>goals(num_robots);
        for (size_t rob_idx = 0; rob_idx < num_robots; rob_idx++) {
            starts[rob_idx] = start_q.segment(spit, rob_dim);
            goals[rob_idx] = goal_q.segment(spit, rob_dim);
            spit += rob_dim;
        }
        LOG(INFO) << "\n\nRunning Case " << case_id << "...";

        auto problem = std::make_shared<ilqgames::MultiRobotEval>(
            starts, goals,
            base_trans, base_quats,
            env_configs,
            is_floor_on, is_box_on, is_move_on
        );
        problem->Initialize();

        if (robots_for_viz.empty()) {
                robots_for_viz = problem->get_robots();
                env = problem->get_sdf();
                moving_obs = problem->get_moving_obs();
        }

        ilqgames::AugmentedLagrangianSolverDouble solver(problem, params);

        // Solve
        auto start_time = std::chrono::system_clock::now();
        auto log = solver.Solve();
        double runtime = std::chrono::duration<double>(std::chrono::system_clock::now() - start_time).count();
        LOG(INFO) << "Solver finished in " << runtime << "s";

        // verify solution
        bool is_success = recorder.RecordMatch(case_id, runtime, log, problem, starts, goals, robots_for_viz);

        // Store log for visualization
        all_problems_logs.push_back({log});
        all_problems.push_back(problem);

        // Optional: Save individual logs to disk
        if (FLAGS_save) {
        std::string log_name;
        if (is_success)
            log_name = Test_name + "/Success_" + std::to_string(case_id);
        else
            log_name = Test_name + "/Failed_" + std::to_string(case_id);
            log->Save(FLAGS_last_traj, log_name, true);
        }

        case_id++;
    }

    if (FLAGS_save) recorder.SaveSummary(Test_name);
 
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

    // If this file doesn't exist, check /usr/share/fonts/ for other .ttf files.
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16.0f);

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    std::shared_ptr<ilqgames::ControlSlidersDouble> sliders(
        new ilqgames::ControlSlidersDouble(all_problems_logs));

    std::vector<std::vector<ilqgames::PlayerCostDouble>> all_playercosts;
    for(auto& p : all_problems) {
        all_playercosts.push_back(p->PlayerCosts());
    }

    ilqgames::Rob3DRenderer rob_3d_render(sliders, all_problems, robots_for_viz, env, moving_obs);
    ilqgames::CostInspectorDouble cost_inspector(sliders, all_playercosts);

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