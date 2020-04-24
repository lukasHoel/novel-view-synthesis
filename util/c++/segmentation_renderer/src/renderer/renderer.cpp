#include "renderer.h"
#include "model.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = DEF_WIDTH / 2.0f;
float lastY = DEF_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

Renderer::Renderer(string const &path) {
    if(init()){
        // if init fails, then the return code is != 0 which is equal to this if statement
        throw std::runtime_error("Failed to init renderer");
    }

    m_model = new Model(path);
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

void Renderer::renderToImage(const glm::mat4& pose, const glm::mat4& projection, const std::string save_path){
    render(glm::mat4(), pose, projection);

    // Note on the Matterport dataset:

    /*
        - The region_X.ply are still in world-coordinates, e.g. region0 is left and region6 is centered.
        - Thus I can use the camera extrinsics/intrinsics also for the regions only
        - This means, that I can use regions + vseg file (Alternative: use whole house mesh and parse fseg file instead of vseg)
        - For each image (matterport_color_images.zip) we have a corresponding extrinsic/intrinsic file with same name
            --> Use this for calculating the view and projection matrices
            --> But these parameters are distorted, e.g. the intrinsic files contain arbitrary 3x3 matrix
            --> This is solved in undistorted_camera_parameters.zip
            --> The same values as in undistorted_camera_parameters.zip are also present in the .house file
            --> Just use the extrinsic/intrinsic parameters from the .house file!
            --> Note that the extrinsic parameters differ in the .house file and in the undistorted file. What is correct?
            --> TODO: HOW?
        - TODO: Find out which image corresponds to which region. It only makes sense to use the images for the corresponding region
            --> Otherwise we would look at nothing because in that case the region is not present
            --> Can I do it like this? Parse .house file and go like this: Image Name --> Panorama Index --> Region Index ?
        - TODO: Write a main.cpp pipeline which loops over all images (selected images --> loop over folder of images) for a specific region
            --> Color according to segmentation + transform object from input
            --> Render from specific camera pose/intrinsic for this view
            --> Save as image in output folder
        - TODO (optional): Before the main.cpp pipeline starts, we show the region in an interactive renderer.
            --> Allow the user to somehow interactively choose which object to move and how to move it
            --> From this selection, extract a transformation matrix and use that as an input for the pipeline
            --> (optional 2): Let the user create a trajectory (multiple transformation matrices) and use each of them
    */

    //TODO get image from openGL and save...
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
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)DEF_WIDTH / (float)DEF_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f)); // translate it down so it's at the center of the scene
        // model = glm::scale(model, glm::vec3(0.2f, 0.2f, 0.2f));	// it's a bit too big for our scene, so scale it down

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