#pragma once

#include <GL/glew.h>

#include <GLFW/glfw3.h>

#include "model.h"
#include "camera.h"

const unsigned int DEF_WIDTH = 800;
const unsigned int DEF_HEIGHT = 600;

class Renderer {
public:
    Renderer(string const &path);
    ~Renderer();
    void renderToImage(const glm::mat4& pose, const glm::mat4& projection, const std::string save_path = "");
    void renderInteractive();
    int init();

    Model* m_model = nullptr;
private:

    int m_buffer_width = DEF_WIDTH;
    int m_buffer_height = DEF_HEIGHT;
    bool m_initialized = false;

    GLFWwindow* m_window = nullptr;
    Shader* m_shader = nullptr;

    void render(const glm::mat4& model, const glm::mat4& view, const glm::mat4& projection);
};