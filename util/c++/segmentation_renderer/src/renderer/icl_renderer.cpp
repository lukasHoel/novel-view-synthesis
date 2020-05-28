#include "icl_renderer.h"
#include "util.h"

ICL_Renderer::ICL_Renderer(string const &pathToMesh) : Renderer(pathToMesh, 640, 480) {   
    // nothing to do here
}

ICL_Renderer::~ICL_Renderer() = default;

void ICL_Renderer::renderTrajectory(ICL_Parser& ip, const std::string save_path){
    
    for(int i=0; i<ip.getNumberofPoseFiles(); i++){
        //glm::mat4 extr = glm::inverse(ip.getExtrinsics(i));
        glm::mat4 extr = ip.getExtrinsics(i);
        glm::mat3 intr = ip.getIntrinsics();

        extr = glm::inverse(extr);
        //intr = glm::inverse(intr);

        //extr = glm::transpose(extr);
        //intr = glm::transpose(intr);

        glm::mat4 projection = camera_utils::perspective(intr, ip.getWidth(), ip.getHeight(), kNearPlane, kFarPlane);

        // render image

        /*
        glm::mat4 model(0.1971501, 0.8028499, 0.5626401, 0,
                  0.8028499, 0.1971501, -0.5626401, 0,
                  -0.5626401, 0.5626401,  -0.6056999, 0,
                  0, 0, 0, 1);
        model = glm::transpose(model); // I have written it in row major but glm uses column major... I hate this stuff.
        */

        std::cout << glm::to_string(extr) << std::endl;

        glm::mat4 trans = glm::mat4(1.0f);
        //trans = glm::translate(trans, glm::vec3(-1.3705f, -1.51739f, -1.44963f));
        //trans = glm::translate(trans, glm::vec3(-0.108f, -1.3f, -2.814f));
        

        // THIS ONE FROM PAPER
        //trans = glm::rotate(trans, glm::radians(180.0f), glm::vec3(0.0, 0.0, 1.0));
        trans = glm::scale(trans, glm::vec3(1, 1, -1));



        //trans = glm::rotate(trans, glm::radians(270.0f), glm::vec3(1.0, 0.0, 0.0));
        //trans = glm::rotate(trans, glm::radians(90.0f), glm::vec3(0.0, 0.0, 1.0));
        
        //trans = glm::rotate(trans, glm::radians(30.0f), glm::vec3(0.0, 0.0, 1.0));
        //trans = glm::translate(trans, glm::vec3(0.3f, 0.3f, 0.3f));
        

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