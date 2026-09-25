#include <ilqgames/gui/rob_3d_renderer.h>
#include <ilqgames/utils/operating_point_double.h>
#include <ilqgames/utils/solver_log_double.h>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>

namespace ilqgames {

// --- Shaders ---
static const char* ROBOT_VS = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    out vec3 FragPos;
    out vec3 Normal;
    void main() {
        FragPos = vec3(model * vec4(aPos, 1.0));
        Normal = mat3(transpose(inverse(model))) * aNormal;
        gl_Position = projection * view * vec4(FragPos, 1.0);
    }
)";

static const char* ROBOT_FS = R"(
    #version 330 core
    out vec4 FragColor;
    in vec3 FragPos;
    in vec3 Normal;
    uniform vec4 color; 
    uniform vec3 lightPos;
    void main() {
        float ambientStrength = 0.4;
        vec3 ambient = ambientStrength * vec3(1.0);
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * vec3(1.0);
        
        vec3 result = (ambient + diffuse) * color.rgb;
        FragColor = vec4(result, color.a);
    }
)";

static unsigned int sphere_vao_ = 0, sphere_vbo_ = 0, sphere_count_ = 0;
static unsigned int cyl_vao_ = 0, cyl_vbo_ = 0, cyl_count_ = 0;

void Rob3DRenderer::InitSphereGeometry() {
    std::vector<float> verts;
    const int stacks = 12;
    const int slices = 12;
    for (int i = 0; i < stacks; ++i) {
        float phi1 = M_PI * float(i) / float(stacks);
        float phi2 = M_PI * float(i + 1) / float(stacks);
        for (int j = 0; j < slices; ++j) {
            float theta1 = 2.0f * M_PI * float(j) / float(slices);
            float theta2 = 2.0f * M_PI * float(j + 1) / float(slices);
            auto push_vert = [&](float phi, float theta) {
                float x = sin(phi) * cos(theta);
                float y = sin(phi) * sin(theta);
                float z = cos(phi);
                verts.push_back(x); verts.push_back(y); verts.push_back(z);
                verts.push_back(x); verts.push_back(y); verts.push_back(z);
            };
            push_vert(phi1, theta1); push_vert(phi2, theta1); push_vert(phi2, theta2);
            push_vert(phi1, theta1); push_vert(phi2, theta2); push_vert(phi1, theta2);
        }
    }
    sphere_count_ = verts.size() / 6;
    glGenVertexArrays(1, &sphere_vao_);
    glGenBuffers(1, &sphere_vbo_);
    glBindVertexArray(sphere_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, sphere_vbo_);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float))); glEnableVertexAttribArray(1);
}

void Rob3DRenderer::InitCylinderGeometry() {
    std::vector<float> verts;
    const int slices = 12;
    for (int i = 0; i < slices; ++i) {
        float theta1 = 2.0f * M_PI * float(i) / float(slices);
        float theta2 = 2.0f * M_PI * float(i + 1) / float(slices);
        float x1 = cos(theta1), y1 = sin(theta1);
        float x2 = cos(theta2), y2 = sin(theta2);
        float v[6][6] = {
            {x1, y1, -0.5f, x1, y1, 0.0f}, {x2, y2, -0.5f, x2, y2, 0.0f}, {x1, y1, 0.5f, x1, y1, 0.0f},
            {x2, y2, -0.5f, x2, y2, 0.0f}, {x2, y2, 0.5f, x2, y2, 0.0f}, {x1, y1, 0.5f, x1, y1, 0.0f}
        };
        for(auto& row : v) for(float val : row) verts.push_back(val);
    }
    cyl_count_ = verts.size() / 6;
    glGenVertexArrays(1, &cyl_vao_);
    glGenBuffers(1, &cyl_vbo_);
    glBindVertexArray(cyl_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, cyl_vbo_);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float))); glEnableVertexAttribArray(1);
}

void Rob3DRenderer::InitAxis() {
    float axisVertices[] = {
        0.0f, 0.0f, 0.0f,  1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,  0.0f, 0.0f, 1.0f
    };
    glGenVertexArrays(1, &axis_vao_);
    glGenBuffers(1, &axis_vbo_);
    glBindVertexArray(axis_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, axis_vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(axisVertices), axisVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glBindVertexArray(0);
}

void Rob3DRenderer::InitCubeGeometry() {
    std::vector<float> verts;
    auto addQuad = [&](Eigen::Vector3f p1, Eigen::Vector3f p2, Eigen::Vector3f p3, Eigen::Vector3f p4, Eigen::Vector3f n) {
        verts.insert(verts.end(), {p1.x(), p1.y(), p1.z(), n.x(), n.y(), n.z()});
        verts.insert(verts.end(), {p2.x(), p2.y(), p2.z(), n.x(), n.y(), n.z()});
        verts.insert(verts.end(), {p3.x(), p3.y(), p3.z(), n.x(), n.y(), n.z()});
        verts.insert(verts.end(), {p1.x(), p1.y(), p1.z(), n.x(), n.y(), n.z()});
        verts.insert(verts.end(), {p3.x(), p3.y(), p3.z(), n.x(), n.y(), n.z()});
        verts.insert(verts.end(), {p4.x(), p4.y(), p4.z(), n.x(), n.y(), n.z()});
    };
    Eigen::Vector3f p000(0,0,0), p100(1,0,0), p110(1,1,0), p010(0,1,0);
    Eigen::Vector3f p001(0,0,1), p101(1,0,1), p111(1,1,1), p011(0,1,1);
    
    // Explicit float normals
    addQuad(p001, p101, p111, p011, Eigen::Vector3f(0,0,1));
    addQuad(p100, p000, p010, p110, Eigen::Vector3f(0,0,-1));
    addQuad(p101, p100, p110, p111, Eigen::Vector3f(1,0,0));
    addQuad(p000, p001, p011, p010, Eigen::Vector3f(-1,0,0));
    addQuad(p011, p111, p110, p010, Eigen::Vector3f(0,1,0));
    addQuad(p000, p100, p101, p001, Eigen::Vector3f(0,-1,0));
    
    cube_count_ = verts.size() / 6;
    glGenVertexArrays(1, &cube_vao_);
    glGenBuffers(1, &cube_vbo_);
    glBindVertexArray(cube_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, cube_vbo_);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float))); glEnableVertexAttribArray(1);
}

void Rob3DRenderer::InitSDFMeshes() {
    sdf_gpu_meshes_.clear();

    for (const auto& sdf : sdfs_) {
        std::vector<float> batch_verts;
        batch_verts.reserve(100000); 

        // Get SDF parameters
        Eigen::Vector3d origin = sdf->GetOrigin();
        Eigen::Vector3d res = sdf->GetResolution();
        Eigen::Vector3i dim = sdf->GetDim();

        auto is_solid = [&](int x, int y, int z) {
            if (x < 0 || x >= dim.x() || y < 0 || y >= dim.y() || z < 0 || z >= dim.z()) 
                return false;
            
            // Calculate world position center of this voxel
            Eigen::Vector3d p(origin.x() + x * res.x(), 
                              origin.y() + y * res.y(), 
                              origin.z() + z * res.z());
            
            return sdf->GetDistance(p) <= 0.0;
        };

        auto add_face = [&](int x, int y, int z, 
                            const float* c1, const float* c2, const float* c3, const float* c4, 
                            float nx, float ny, float nz) {
            
            float ox = (float)origin.x() + x * (float)res.x();
            float oy = (float)origin.y() + y * (float)res.y();
            float oz = (float)origin.z() + z * (float)res.z();
            
            float rx = (float)res.x();
            float ry = (float)res.y();
            float rz = (float)res.z();

            auto push_vert = [&](const float* c) {
                batch_verts.push_back(ox + c[0]*rx); // Pos X
                batch_verts.push_back(oy + c[1]*ry); // Pos Y
                batch_verts.push_back(oz + c[2]*rz); // Pos Z
                batch_verts.push_back(nx);           // Norm X
                batch_verts.push_back(ny);           // Norm Y
                batch_verts.push_back(nz);           // Norm Z
            };

            push_vert(c1); push_vert(c2); push_vert(c3);
            push_vert(c1); push_vert(c3); push_vert(c4);
        };

        // Offsets for centered voxels
        const float p000[] = {-0.5f,-0.5f,-0.5f}; const float p100[] = {0.5f,-0.5f,-0.5f};
        const float p010[] = {-0.5f, 0.5f,-0.5f}; const float p110[] = {0.5f, 0.5f,-0.5f};
        const float p001[] = {-0.5f,-0.5f, 0.5f}; const float p101[] = {0.5f,-0.5f, 0.5f};
        const float p011[] = {-0.5f, 0.5f, 0.5f}; const float p111[] = {0.5f, 0.5f, 0.5f};
        
        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                for (int x = 0; x < dim.x(); ++x) {
                    if (is_solid(x, y, z)) {
                        if (!is_solid(x + 1, y, z)) add_face(x, y, z, p101, p100, p110, p111, 1, 0, 0);
                        if (!is_solid(x - 1, y, z)) add_face(x, y, z, p000, p001, p011, p010, -1, 0, 0);
                        if (!is_solid(x, y + 1, z)) add_face(x, y, z, p011, p111, p110, p010, 0, 1, 0);
                        if (!is_solid(x, y - 1, z)) add_face(x, y, z, p000, p100, p101, p001, 0, -1, 0);
                        if (!is_solid(x, y, z + 1)) add_face(x, y, z, p001, p101, p111, p011, 0, 0, 1);
                        if (!is_solid(x, y, z - 1)) add_face(x, y, z, p100, p000, p010, p110, 0, 0, -1);
                    }
                }
            }
        }

        if (!batch_verts.empty()) {
            GPUMesh mesh;
            mesh.vertexCount = batch_verts.size() / 6;
            mesh.color = {0.9f, 0.9f, 0.9f}; 
            glGenVertexArrays(1, &mesh.VAO);
            glGenBuffers(1, &mesh.VBO);
            glBindVertexArray(mesh.VAO);
            glBindBuffer(GL_ARRAY_BUFFER, mesh.VBO);
            glBufferData(GL_ARRAY_BUFFER, batch_verts.size() * sizeof(float), batch_verts.data(), GL_STATIC_DRAW);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0); glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float))); glEnableVertexAttribArray(1);
            sdf_gpu_meshes_.push_back(mesh);
        }
    }
}

void Rob3DRenderer::DrawCapsules(const std::vector<Robot*>& robots, unsigned int shader) {
    glUniform4f(glGetUniformLocation(shader, "color"), 1.0f, 0.0f, 0.8f, 1.0f);
    for (const auto* robot : robots) {
        const auto& capsules = robot->get_capsules();
        for (const auto& cap : capsules) {
            Eigen::Vector3d p1 = cap.p1_world;
            Eigen::Vector3d p2 = cap.p2_world;
            float r = (float)cap.radius_;
            Eigen::Matrix4f s_model = Eigen::Matrix4f::Identity();
            s_model(0,0) = r; s_model(1,1) = r; s_model(2,2) = r;
            s_model.block<3,1>(0,3) = p1.cast<float>();
            glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, s_model.data());
            glBindVertexArray(sphere_vao_);
            glDrawArrays(GL_TRIANGLES, 0, sphere_count_);
            s_model.block<3,1>(0,3) = p2.cast<float>();
            glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, s_model.data());
            glDrawArrays(GL_TRIANGLES, 0, sphere_count_);
            Eigen::Vector3d diff = p2 - p1;
            double len = diff.norm();
            if (len > 1e-5) {
                Eigen::Vector3d center = (p1 + p2) * 0.5;
                Eigen::Vector3d z_axis(0,0,1);
                Eigen::Quaterniond q;
                q.setFromTwoVectors(z_axis, diff.normalized());
                Eigen::Matrix4f c_model = Eigen::Matrix4f::Identity();
                c_model.block<3,3>(0,0) = q.toRotationMatrix().cast<float>(); 
                c_model.block<3,1>(0,3) = center.cast<float>();            
                Eigen::Matrix4f scale = Eigen::Matrix4f::Identity();
                scale(0,0) = r; scale(1,1) = r; scale(2,2) = (float)len;
                c_model = c_model * scale;
                glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, c_model.data());
                glBindVertexArray(cyl_vao_);
                glDrawArrays(GL_TRIANGLES, 0, cyl_count_);
            }
        }
    }
}

void Rob3DRenderer::DrawSDFs(unsigned int shader) {
    if (sdf_gpu_meshes_.empty()) return;
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, model.data());
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUniform4f(glGetUniformLocation(shader, "color"), 1.0f, 1.0f, 1.0f, 0.5f);
    for (const auto& mesh : sdf_gpu_meshes_) {
        glBindVertexArray(mesh.VAO);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)mesh.vertexCount);
    }
    glDisable(GL_BLEND);
    glBindVertexArray(0);
}

void Rob3DRenderer::DrawMovingObstacles(double t, unsigned int shader) {
    if (moving_obstacles_.empty()) return;

    glBindVertexArray(cube_vao_);
    glUniform4f(glGetUniformLocation(shader, "color"), 0.2f, 0.9f, 0.2f, 1.0f); // Green color for moving obstacles

    for (const auto& obs : moving_obstacles_) {
        Eigen::Vector3d pos = obs.GetPosition(t);
        Eigen::Vector3d corner_d = pos - obs.size * 0.5;
        Eigen::Vector3f corner_f = corner_d.cast<float>();
        Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
        model.block<3,1>(0,3) = corner_f;
        
        // Scale
        Eigen::Matrix4f scale = Eigen::Matrix4f::Identity();
        scale(0,0) = static_cast<float>(obs.size.x());
        scale(1,1) = static_cast<float>(obs.size.y());
        scale(2,2) = static_cast<float>(obs.size.z());

        model = model * scale;

        glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, model.data());
        glDrawArrays(GL_TRIANGLES, 0, cube_count_);
    }
    glBindVertexArray(0);
}

void Rob3DRenderer::DrawAxes(const Eigen::Matrix4f& transform, float scale) {
    glBindVertexArray(axis_vao_);
    Eigen::Matrix4f model = transform;
    Eigen::Matrix4f scaleMat = Eigen::Matrix4f::Identity();
    scaleMat(0,0) = scale; scaleMat(1,1) = scale; scaleMat(2,2) = scale;
    model = model * scaleMat;
    glUniformMatrix4fv(glGetUniformLocation(shader_, "model"), 1, GL_FALSE, model.data());
    glUniform4f(glGetUniformLocation(shader_, "color"), 1.0f, 0.0f, 0.0f, 1.0f); glDrawArrays(GL_LINES, 0, 2);
    glUniform4f(glGetUniformLocation(shader_, "color"), 0.0f, 1.0f, 0.0f, 1.0f); glDrawArrays(GL_LINES, 2, 2);
    glUniform4f(glGetUniformLocation(shader_, "color"), 0.0f, 0.0f, 1.0f, 1.0f); glDrawArrays(GL_LINES, 4, 2);
    glBindVertexArray(0);
}

Rob3DRenderer::Rob3DRenderer(
      const std::shared_ptr<const ControlSlidersDouble>& sliders,
      const std::vector<std::shared_ptr<const TopDownRobRenderableProblem>>& problems,
      const std::vector<Robot*> robots,
      const std::vector<std::shared_ptr<SignedDistanceField>>& sdfs,
      const std::vector<MovingObs>& moving_obstacles)
    : sliders_(sliders), problems_(problems), robots_(robots), sdfs_(sdfs),
      moving_obstacles_(moving_obstacles)
    {
        InitOpenGL();
        LoadMeshes();
    }

Rob3DRenderer::~Rob3DRenderer() {
    if(fbo_) glDeleteFramebuffers(1, &fbo_);
    if(texId_) glDeleteTextures(1, &texId_);
    if(rbo_) glDeleteRenderbuffers(1, &rbo_);
    if(shader_) glDeleteProgram(shader_);
    if(axis_vao_) glDeleteVertexArrays(1, &axis_vao_);
    if(axis_vbo_) glDeleteBuffers(1, &axis_vbo_);
    if(cube_vao_) glDeleteVertexArrays(1, &cube_vao_);
    if(cube_vbo_) glDeleteBuffers(1, &cube_vbo_);

    for(auto& m : meshes_) {
        if(m.VAO) glDeleteVertexArrays(1, &m.VAO);
        if(m.VBO) glDeleteBuffers(1, &m.VBO);
    }
    for(auto& m : sdf_gpu_meshes_) {
        if(m.VAO) glDeleteVertexArrays(1, &m.VAO);
        if(m.VBO) glDeleteBuffers(1, &m.VBO);
    }
}

void Rob3DRenderer::Render() {
    UpdateKinematics();
    RenderToFBO();
    ImGui::Begin("3D Visualization with Pinocchio");
    ImVec2 size = ImGui::GetContentRegionAvail();
    if (size.x > 0 && size.y > 0 && (size.x != viewport_w_ || size.y != viewport_h_)) {
        ResizeFBO((int)size.x, (int)size.y);
    }
    ImGui::Image((void*)(intptr_t)texId_, size, ImVec2(0, 1), ImVec2(1, 0));
    if (ImGui::IsItemHovered()) {
        UpdateCamera();
    }
    ImGui::End();
}

void Rob3DRenderer::InitOpenGL() {
    unsigned int v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, &ROBOT_VS, NULL); glCompileShader(v);
    unsigned int f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &ROBOT_FS, NULL); glCompileShader(f);
    shader_ = glCreateProgram();
    glAttachShader(shader_, v); glAttachShader(shader_, f);
    glLinkProgram(shader_);
    glDeleteShader(v); glDeleteShader(f);

    ResizeFBO(viewport_w_, viewport_h_);
    InitAxis();
    InitSphereGeometry();
    InitCylinderGeometry();
    InitCubeGeometry();
    InitSDFMeshes();
}

void Rob3DRenderer::ResizeFBO(int w, int h) {
    if (w <= 0 || h <= 0) return;
    viewport_w_ = w; viewport_h_ = h;
    if (fbo_) {
        glDeleteFramebuffers(1, &fbo_);
        glDeleteTextures(1, &texId_);
        glDeleteRenderbuffers(1, &rbo_);
    }
    glGenFramebuffers(1, &fbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glGenTextures(1, &texId_);
    glBindTexture(GL_TEXTURE_2D, texId_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texId_, 0);
    glGenRenderbuffers(1, &rbo_);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo_);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Rob3DRenderer::LoadMeshes() {
    meshes_.clear();
    for(size_t r_idx = 0; r_idx < robots_.size(); ++r_idx) {
        const auto& robot = robots_[r_idx];
        const auto& geomModel = robot->get_geom_model();
        for(size_t g_idx = 0; g_idx < geomModel.geometryObjects.size(); ++g_idx) {
            const auto& go = geomModel.geometryObjects[g_idx];
            GPUMesh mesh;
            mesh.robot_idx = r_idx;
            mesh.geom_idx = g_idx;
            if (go.name.find("link") != std::string::npos) mesh.color = {0.8f, 0.4f, 0.2f};
            else if (go.name.find("base") != std::string::npos) mesh.color = {0.2f, 0.2f, 0.2f};
            else mesh.color = {0.7f, 0.7f, 0.7f};
            std::vector<float> verts;
            if(LoadSTL(go.meshPath, verts)) {
                mesh.vertexCount = verts.size() / 6;
                glGenVertexArrays(1, &mesh.VAO);
                glGenBuffers(1, &mesh.VBO);
                glBindVertexArray(mesh.VAO);
                glBindBuffer(GL_ARRAY_BUFFER, mesh.VBO);
                glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_STATIC_DRAW);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)0); glEnableVertexAttribArray(0);
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)(3*sizeof(float))); glEnableVertexAttribArray(1);
                meshes_.push_back(mesh);
            }
        }
    }
}

bool Rob3DRenderer::LoadSTL(const std::string& path, std::vector<float>& buffer) {
    std::ifstream file(path, std::ios::binary);
    if(!file) return false;
    char header[80]; file.read(header, 80);
    std::string hStr(header, 5);
    if(hStr != "solid") { 
        uint32_t count; file.read((char*)&count, 4);
        for(uint32_t i=0; i<count; ++i) {
            float n[3], v[3][3];
            uint16_t attr;
            file.read((char*)n, 12);
            file.read((char*)v[0], 12); file.read((char*)v[1], 12); file.read((char*)v[2], 12);
            file.read((char*)&attr, 2);
            for(int k=0; k<3; ++k) {
                buffer.push_back(v[k][0]); buffer.push_back(v[k][1]); buffer.push_back(v[k][2]);
                buffer.push_back(n[0]); buffer.push_back(n[1]); buffer.push_back(n[2]);
            }
        }
    } else { 
        file.close(); file.open(path);
        std::string t; float n[3]={0}, v[3];
        while(file >> t) {
            if(t == "facet") { file >> t >> n[0] >> n[1] >> n[2]; }
            else if(t == "vertex") {
                file >> v[0] >> v[1] >> v[2];
                buffer.push_back(v[0]); buffer.push_back(v[1]); buffer.push_back(v[2]);
                buffer.push_back(n[0]); buffer.push_back(n[1]); buffer.push_back(n[2]);
            }
        }
    }
    return true;
}

void Rob3DRenderer::UpdateKinematics() {
    const auto& logs = sliders_->LogForEachProblem();
    const size_t problem_idx = sliders_->ProbIndex();
    const auto& problem = problems_[problem_idx];
    const auto& log = logs[problem_idx];
    ilqgames::Time t = sliders_->InterpolationTime(problem_idx);
    int iterate = sliders_->SolverIterate(problem_idx);
    VectorXd x = log->InterpolateState(iterate, t);
    std::vector<double> all_qs = problem->Qs(x);
    int offset = 0;
    for (auto& robot : robots_) {
        int nq = robot->get_qDim();
        if (offset + nq > (int)all_qs.size()) break;
        Eigen::Map<Eigen::VectorXd> q_sub(all_qs.data() + offset, nq);
        robot->updateRobotViz(q_sub);
        robot->updateRobotJoints(q_sub);
        offset += nq;
    }
}

void Rob3DRenderer::UpdateCamera() {
    ImGuiIO& io = ImGui::GetIO();
    if(ImGui::IsMouseDown(1)) {
        ImVec2 d = ImGui::GetMouseDragDelta(1);
        cam_yaw_ -= d.x * 0.005f;
        cam_pitch_ += d.y * 0.005f;
        ImGui::ResetMouseDragDelta(1);
    }
    cam_dist_ -= io.MouseWheel * 0.2f;
    if(cam_dist_ < 0.1f) cam_dist_ = 0.1f;
}

void Rob3DRenderer::RenderToFBO() {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glViewport(0, 0, viewport_w_, viewport_h_);
    glClearColor(0.2f, 0.2f, 0.25f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glUseProgram(shader_);

    Eigen::Vector3f eye;
    eye.x() = cam_target_.x() + cam_dist_ * std::cos(cam_pitch_) * std::cos(cam_yaw_);
    eye.y() = cam_target_.y() + cam_dist_ * std::cos(cam_pitch_) * std::sin(cam_yaw_);
    eye.z() = cam_target_.z() + cam_dist_ * std::sin(cam_pitch_);
    Eigen::Vector3f f = (cam_target_ - eye).normalized();
    Eigen::Vector3f s = f.cross(Eigen::Vector3f(0,0,1)).normalized();
    Eigen::Vector3f u = s.cross(f);
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
    view(0,0)=s.x(); view(0,1)=s.y(); view(0,2)=s.z(); view(0,3)=-s.dot(eye);
    view(1,0)=u.x(); view(1,1)=u.y(); view(1,2)=u.z(); view(1,3)=-u.dot(eye);
    view(2,0)=-f.x(); view(2,1)=-f.y(); view(2,2)=-f.z(); view(2,3)=f.dot(eye);
    float aspect = (float)viewport_w_ / (float)viewport_h_;
    float fovy = 0.785f;
    float tanHalf = std::tan(fovy/2.0f);
    Eigen::Matrix4f proj = Eigen::Matrix4f::Zero();
    proj(0,0) = 1.0f / (aspect * tanHalf);
    proj(1,1) = 1.0f / tanHalf;
    proj(2,2) = -(100.0f + 0.1f) / (100.0f - 0.1f);
    proj(2,3) = -(2.0f * 100.0f * 0.1f) / (100.0f - 0.1f);
    proj(3,2) = -1.0f;

    glUniformMatrix4fv(glGetUniformLocation(shader_, "view"), 1, GL_FALSE, view.data());
    glUniformMatrix4fv(glGetUniformLocation(shader_, "projection"), 1, GL_FALSE, proj.data());
    glUniform3f(glGetUniformLocation(shader_, "lightPos"), 2.0f, 2.0f, 5.0f);

    DrawAxes(Eigen::Matrix4f::Identity(), 0.5f);
    DrawSDFs(shader_); 

    double t = 0.0;
    if (sliders_->ProbIndex() < problems_.size()) {
        t = sliders_->InterpolationTime(sliders_->ProbIndex());
    }
    DrawMovingObstacles(t, shader_);

    for(const auto& mesh : meshes_) {
        if(mesh.vertexCount == 0) continue;
        const auto& robot = robots_[mesh.robot_idx];
        const auto& geomData = robot->get_geom_data();
        const auto& geomModel = robot->get_geom_model();
        Eigen::Matrix4f local_model = geomData.oMg[mesh.geom_idx].toHomogeneousMatrix().cast<float>();
        Eigen::Matrix4f world_base = robot->get_base().toHomogeneousMatrix().cast<float>();
        Eigen::Matrix4f model = world_base * local_model;
        Eigen::Vector3d sc = geomModel.geometryObjects[mesh.geom_idx].meshScale;
        Eigen::Matrix4f scale = Eigen::Matrix4f::Identity();
        scale(0,0)=sc[0]; scale(1,1)=sc[1]; scale(2,2)=sc[2];
        model = model * scale;
        glUniformMatrix4fv(glGetUniformLocation(shader_, "model"), 1, GL_FALSE, model.data());
        glUniform4f(glGetUniformLocation(shader_, "color"), mesh.color[0], mesh.color[1], mesh.color[2], 0.4f);
        glBindVertexArray(mesh.VAO);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)mesh.vertexCount);
    }
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE); 
    DrawCapsules(robots_, shader_);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

} // namespace ilqgames