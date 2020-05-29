#include "renderer.h"
#include "model.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

// camera
Camera camera(glm::vec3(0.790932f, 1.300000f, 1.462270f)); // 1.3705f, 1.51739f, 1.44963f    0.0f, 0.0f, 3.0f      -0.3f, 0.3f, 0.3f
float lastX = DEF_WIDTH / 2.0f;
float lastY = DEF_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

Renderer::Renderer(string const &pathToMesh, int width, int height) {  
    m_buffer_width = width;
    m_buffer_height = height;

    if(init()){
        // if init fails, then the return code is != 0 which is equal to this if statement
        throw std::runtime_error("Failed to init renderer");
    }

    m_model = new Model(pathToMesh);
    m_shader = new Shader("../shader/color3D.vs", "../shader/color3D.frag");
}

Renderer::~Renderer() {
    delete &m_model;
    delete &m_shader;
    glfwTerminate();
}

int Renderer::init() {
    if(! glfwInit()){
        std::cout << "Failed to init glfw" << std::endl;
        return EXIT_FAILURE;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

    // glfw window creation
    // --------------------
    m_window = glfwCreateWindow(m_buffer_width, m_buffer_height, "Segmentation_Renderer", NULL, NULL);
    if (m_window == nullptr) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return EXIT_FAILURE;
    }
    glfwMakeContextCurrent(m_window);

    // To avoid: https://stackoverflow.com/questions/8302625/segmentation-fault-at-glgenvertexarrays-1-vao
    glewExperimental = GL_TRUE; 
    if (GLEW_OK != glewInit()){
        std::cout << "Failed to init glew" << std::endl;
        return EXIT_FAILURE;
    }

    glfwSetFramebufferSizeCallback(m_window, framebuffer_size_callback);
    glfwSetCursorPosCallback(m_window, mouse_callback);
    glfwSetScrollCallback(m_window, scroll_callback);
    // tell GLFW to capture our mouse
    glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // configure global opengl state
    glEnable(GL_DEPTH_TEST);

    m_initialized = true;
    return 0;
}

void Renderer::render(const glm::mat4& model, const glm::mat4& view, const glm::mat4& projection){
    if(! m_initialized){
        std::cout << "Cannot render before initializing the renderer" << std::endl;
        return;
    }

    GLuint framebuffername = 1;
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffername);

    glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_shader->use();

    m_shader->setMat4("projection", projection);
    m_shader->setMat4("view", view);
    m_shader->setMat4("model", model);

    m_model->draw(*m_shader);
}

void Renderer::readRGB(cv::Mat& image) {
    glBindFramebuffer(GL_FRAMEBUFFER, 4);
    image = cv::Mat(m_buffer_height, m_buffer_width, CV_8UC3);
    std::vector<float> data_buff(m_buffer_width * m_buffer_height * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, m_buffer_width, m_buffer_height, GL_RGB, GL_FLOAT, data_buff.data());
    for (int i = 0; i < m_buffer_height; ++i) {
        for (int j = 0; j < m_buffer_width; ++j) {
            for (int c = 0; c < 3; c++) {
                image.at<cv::Vec3b>(m_buffer_height - i - 1, j)[2 - c] = 
                    static_cast<int>(256 * data_buff[int(3 * i * m_buffer_width + 3 * j + c)]);
            }
        }
    }
    // cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
}

void Renderer::renderInteractive(){
    // render loop
    while (!glfwWindowShouldClose(m_window))
    {

        // per-frame time logic
        // --------------------
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----
        processInput(m_window);
        
        // model/view/projection transformations
        // ------
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)m_buffer_width / (float)m_buffer_height, 0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();
        glm::mat4 model = glm::mat4(1.0f);

        //model = glm::scale(model, glm::vec3(1, 1, -1));
        //model = glm::rotate(model, glm::radians(180.0f), glm::vec3(0.0, 0.0, 1.0));
        //model = glm::rotate(model, glm::radians(-90.0f), glm::vec3(1.0, 0.0, 0.0));

        //model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f)); // translate it down so it's at the center of the scene
        //model = glm::scale(model, glm::vec3(0.2f, 0.2f, 0.2f));	// it's a bit too big for our scene, so scale it down

        // render
        // ------
        render(model, view, projection);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(m_window);
        glfwPollEvents();
    }
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(yoffset);
}