#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <opencv2/opencv.hpp>

#include "model.h"
#include "camera.h"
#include "mp.h"

const unsigned int DEF_WIDTH = 1280;
const unsigned int DEF_HEIGHT = 1024;

constexpr float kNearPlane{0.1f};
constexpr float kFarPlane{10.0f};

class Renderer {
public:
    Renderer(string const &pathToMesh, MP_Parser const &mp_parser, int region_index);
    ~Renderer();
    void renderImages(const std::string save_path = "");
    void renderInteractive();
    int init();

    Model* m_model = nullptr;
private:

    int m_buffer_width = DEF_WIDTH;
    int m_buffer_height = DEF_HEIGHT;
    bool m_initialized = false;
    
    MP_Parser mp_parser;
    int region_index = -1;

    GLFWwindow* m_window = nullptr;
    Shader* m_shader = nullptr;

    void render(const glm::mat4& model, const glm::mat4& view, const glm::mat4& projection);
    void readRGB(cv::Mat& image);
};