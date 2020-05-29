#include "icl_renderer.h"
#include "util.h"

ICL_Renderer::ICL_Renderer(string const &pathToMesh) : Renderer(pathToMesh, 640, 480) {   
    // nothing to do here
}

ICL_Renderer::~ICL_Renderer() = default;

void ICL_Renderer::renderTrajectory(ICL_Parser& ip, const std::string save_path){
    
    for(int i=0; i<ip.getNumberofPoseFiles(); i++){
        //glm::mat4 extr = ip.getExtrinsics(i);
        glm::mat4 extr = ip.getExtrinsics(i);
        glm::mat3 intr = ip.getIntrinsics();

        extr = glm::inverse(extr); // RT goes from view to world, but we need world to view
        //intr = glm::inverse(intr);

        //extr = glm::transpose(extr); // THIS SHOULD NOT BE NECESSARY, is already in column-mayor from method
        //intr = glm::transpose(intr); // THIS SHOULD NOT BE NECESSARY, is already in column-mayor from method

        glm::mat4 projection = camera_utils::perspective(intr, ip.getWidth(), ip.getHeight(), kNearPlane, kFarPlane);

        // render image

        std::cout << glm::to_string(extr) << std::endl;

        glm::mat4 trans = glm::mat4(1.0f);
        //trans = glm::translate(trans, glm::vec3(0.3f, 0.3f, 0.3f));
        //trans = glm::translate(trans, glm::vec3(-0.108f, -1.3f, -2.814f));
        

        // THIS ONE FROM PAPER
        //trans = glm::rotate(trans, glm::radians(180.0f), glm::vec3(0.0, 0.0, 1.0));

        //trans = glm::rotate(trans, glm::radians(90.0f), glm::vec3(1.0, 0.0, 0.0));
        //trans = glm::scale(trans, glm::vec3(1, 1, -1));
        //trans = glm::rotate(trans, glm::radians(90.0f), glm::vec3(1.0, 0.0, 0.0));
        //trans = glm::rotate(trans, glm::radians(-90.0f), glm::vec3(1.0, 0.0, 0.0));
        //trans = glm::scale(trans, glm::vec3(1, 1, -1));
        //trans = glm::inverse(trans);     

        render(trans, extr, projection);

        // read image into openCV matrix
        cv::Mat colorImage;
        readRGB(colorImage);

        // save matrix as file
        if ((save_path != "") && (!colorImage.empty())) {
            std::stringstream filename;
            char scene_name[30];
            sprintf(scene_name, "scene_%02d_%04d", ip.getSceneNr(), i);
            filename << save_path << "/" << scene_name << ".seg.jpg";
            // cv::imwrite(filename.str(), colorImage);

            std::cout << "Wrote segmentation of: " << scene_name << std::endl;
        }

        // show image in window
        glfwSwapBuffers(m_window);
    
    }
    
}