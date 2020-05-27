#include "icl_renderer.h"
#include "util.h"

ICL_Renderer::ICL_Renderer(string const &pathToMesh) : Renderer(pathToMesh) {   
    // nothing to do here
}

ICL_Renderer::~ICL_Renderer() = default;

void ICL_Renderer::renderTrajectory(ICL_Parser& ip, const std::string save_path){
    
    for(int i=0; i<ip.getNumberofPoseFiles(); i++){
        glm::mat4 extr = ip.getExtrinsics(i);
        glm::mat3 intr = ip.getIntrinsics();
        glm::mat4 projection = camera_utils::perspective(intr, ip.getWidth(), ip.getHeight(), kNearPlane, kFarPlane);

        // render image
        render(glm::mat4(1.0f), extr, projection);

        // read image into openCV matrix
        cv::Mat colorImage;
        readRGB(colorImage);

        // save matrix as file
        if ((save_path != "") && (!colorImage.empty())) {
            std::stringstream filename;
            char scene_name[30];
            sprintf(scene_name, "scene_%02d_%04d", ip.getSceneNr(), i);
            filename << save_path << "/" << scene_name << ".seg.jpg";
            cv::imwrite(filename.str(), colorImage);

            std::cout << "Wrote segmentation of: " << scene_name << std::endl;
        }

        // show image in window
        glfwSwapBuffers(m_window);
    
    }
}