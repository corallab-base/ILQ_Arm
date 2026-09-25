#ifndef ILQGAMES_GUI_ROB_3D_RENDERER_H
#define ILQGAMES_GUI_ROB_3D_RENDERER_H

#define GL_GLEXT_PROTOTYPES 1
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
    #include <GL/gl3w.h>
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
    #include <GL/glew.h>
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
    #include <glad/glad.h>
#else
    #include <GL/gl.h>
    #include <GL/glext.h>
#endif

#include <ilqgames/gui/control_sliders_double.h>
#include <ilqgames/solver/top_down_rob_renderable_problem.h>
#include <ilqgames/utils/types.h>
#include <ilqgames/utils/robot_arm.h> 
#include <ilqgames/utils/signed_distance_field.h>
#include <ilqgames/utils/moving_obstacle.h>

#include <glog/logging.h>
#include <imgui/imgui.h>
#include <Eigen/Core>
#include <vector>
#include <string>
#include <memory>

namespace ilqgames {

struct GPUMesh {
      unsigned int VAO = 0, VBO = 0;
      size_t vertexCount = 0;
      Eigen::Vector3f color = {0.8f, 0.8f, 0.8f};
      size_t robot_idx = 0;
      size_t geom_idx = 0;
};

class Rob3DRenderer {
public:
    Rob3DRenderer(
          const std::shared_ptr<const ControlSlidersDouble>& sliders,
          const std::vector<std::shared_ptr<const TopDownRobRenderableProblem>>& problems,
          const std::vector<Robot*> robots,
          const std::vector<std::shared_ptr<SignedDistanceField>>& sdfs = {},
          const std::vector<MovingObs>& moving_obstacles = {}
    );

    ~Rob3DRenderer();

    void Render();

private:
    void InitOpenGL();
    void LoadMeshes();
    bool LoadSTL(const std::string& path, std::vector<float>& buffer);
    void UpdateKinematics();
    void UpdateCamera();
    void RenderToFBO();
    void ResizeFBO(int w, int h);
    
    void InitAxis();
    void InitSphereGeometry();   // Added Declaration
    void InitCylinderGeometry(); // Added Declaration
    void InitCubeGeometry();
    void InitSDFMeshes();
    
    void DrawAxes(const Eigen::Matrix4f& transform, float scale = 1.0f);
    void DrawSDFs(unsigned int shader);
    void DrawMovingObstacles(double t, unsigned int shader); // <--- ADDED FUNCTION
    void DrawCapsules(const std::vector<Robot*>& robots, unsigned int shader); // Added Declaration

    unsigned int axis_vao_ = 0;
    unsigned int axis_vbo_ = 0;
    
    unsigned int cube_vao_ = 0;
    unsigned int cube_vbo_ = 0;
    unsigned int cube_count_ = 0;

    const std::shared_ptr<const ControlSlidersDouble> sliders_;
    const std::vector<std::shared_ptr<const TopDownRobRenderableProblem>> problems_;
    std::vector<Robot*> robots_;
    std::vector<std::shared_ptr<SignedDistanceField>> sdfs_;
    std::vector<MovingObs> moving_obstacles_; // <--- ADDED MEMBER
    
    std::vector<GPUMesh> sdf_gpu_meshes_; 

    unsigned int fbo_ = 0, texId_ = 0, rbo_ = 0;
    unsigned int shader_ = 0;
    int viewport_w_ = 800, viewport_h_ = 600;
    std::vector<GPUMesh> meshes_;

    float cam_dist_ = 3.0f;
    float cam_yaw_ = 0.785f;
    float cam_pitch_ = 0.5f;
    Eigen::Vector3f cam_target_ = {0.0f, 0.0f, 0.5f};
};

}   // namespace ilqgames

#endif