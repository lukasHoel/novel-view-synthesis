#include "icl_renderer.h"
#include "util.h"

ICL_Renderer::ICL_Renderer(string const &pathToMesh) : Renderer(pathToMesh, 640, 480) {   
    // nothing to do here
}

ICL_Renderer::~ICL_Renderer() = default;

void ICL_Renderer::renderTrajectory(ICL_Parser& ip, const std::string save_path){
    
    for(int i=0; i<ip.getNumberofPoseFiles(); i++){
        glm::mat4 extr = ip.getExtrinsics(i);
        glm::mat3 intr = ip.getIntrinsics();

        //extr = glm::inverse(extr); // RT goes from view to world, but we need world to view
        //intr = glm::inverse(intr);

        //extr = glm::transpose(extr); // THIS SHOULD NOT BE NECESSARY, is already in column-mayor from method
        //intr = glm::transpose(intr); // THIS SHOULD NOT BE NECESSARY, is already in column-mayor from method

        //extr = glm::mat4(1.0f);
        //extr = glm::translate(extr, glm::vec3(1.3705f, 1.51739f, 1.44963f)); // 0.790932, 1.300000, 1.462270         1.3705f, 1.51739f, 1.44963f

        //extr[3][0] *= -1;
        //extr[3][1] *= -1;
        //extr[3][2] *= -1;

        glm::mat4 projection = camera_utils::perspective(intr, ip.getWidth(), ip.getHeight(), kNearPlane, kFarPlane);
        // glm::mat4 projection = glm::mat4(1.0f);

        // render image

        std::cout << glm::to_string(extr) << std::endl;

        glm::mat4 trans = glm::mat4(1.0f);

        trans = glm::scale(trans, glm::vec3(1, -1, 1));
        trans = glm::rotate(trans, glm::radians(180.0f), glm::vec3(0.0, 1.0, 0.0));
        // trans = glm::translate(trans, glm::vec3(-2.75f, 0.0f, 0.0f));




        //trans = glm::rotate(trans, glm::radians(i + 0.0f), glm::vec3(0.0, 1.0, 0.0));
        // trans = glm::rotate(trans, glm::radians(180.0f), glm::vec3(0.0, 0.0, 1.0));
        //trans = glm::scale(trans, glm::vec3(-1, 1, 1));
        //trans = glm::scale(trans, glm::vec3(1, -1, 1));
        //trans = glm::scale(trans, glm::vec3(1, 1, -1));

        // Swap y and z via this matrix
        // glm::mat4 swap = glm::mat4(1.0f);
        // swap[1][1] = 0.0f;
        // swap[2][2] = 0.0f;
        // swap[1][2] = 1.0f;
        // swap[2][1] = 1.0f;



        // trans = glm::rotate(trans, glm::radians(-90.0f), glm::vec3(1.0, 0.0, 0.0));

        // Rot X with all scale combinations
        // trans = glm::rotate(trans, glm::radians(180.0f), glm::vec3(1.0, 0.0, 0.0));
        // trans = glm::scale(trans, glm::vec3(1, 1, -1));

        // trans = glm::rotate(trans, glm::radians(180.0f), glm::vec3(1.0, 0.0, 0.0));
        // trans = glm::scale(trans, glm::vec3(1, -1, 1));

        // trans = glm::rotate(trans, glm::radians(180.0f), glm::vec3(1.0, 0.0, 0.0));
        // trans = glm::scale(trans, glm::vec3(-1, 1, 1));

        // // Rot Y with all scale combinations
        ////////////// CURRENT PREFERRED??? ////////////////////////
        //trans = glm::rotate(trans, glm::radians(180.0f), glm::vec3(0.0, 0.0, 1.0));
        //trans = glm::scale(trans, glm::vec3(1, 1, -1));

        // THIS ONE TOGETHER WITH NON INVERTED EXT AND NON TRANSPOSED EXT ALSO LOOKS GOOD !!!!
        //trans = glm::translate(trans, glm::vec3(0.0f, 0.0f, 0.5f));
        //trans = glm::rotate(trans, glm::radians(180.0f), glm::vec3(0.0, 1.0, 0.0));
        //trans = glm::scale(trans, glm::vec3(1, -1, 1));

        // trans = glm::rotate(trans, glm::radians(180.0f), glm::vec3(0.0, 1.0, 0.0));
        // trans = glm::scale(trans, glm::vec3(1, -1, 1));

        // trans = glm::rotate(trans, glm::radians(180.0f), glm::vec3(0.0, 1.0, 0.0));
        // trans = glm::scale(trans, glm::vec3(-1, 1, 1));

        // // Rot Z with all scale combinations
        // trans = glm::rotate(trans, glm::radians(180.0f), glm::vec3(0.0, 0.0, 1.0));
        // trans = glm::scale(trans, glm::vec3(1, 1, -1));

        // trans = glm::rotate(trans, glm::radians(180.0f), glm::vec3(0.0, 0.0, 1.0));
        // trans = glm::scale(trans, glm::vec3(1, -1, 1));

        // trans = glm::rotate(trans, glm::radians(180.0f), glm::vec3(0.0, 0.0, 1.0));
        // trans = glm::scale(trans, glm::vec3(-1, 1, 1));





        // trans = glm::scale(trans, glm::vec3(1, -1, 1));
        
        // trans = glm::rotate(trans, glm::radians(180.0f), glm::vec3(0.0, 0.0, 1.0));
        //trans = glm::rotate(trans, glm::radians(90.0f), glm::vec3(1.0, 0.0, 0.0));
        // trans = glm::rotate(trans, glm::radians(i + 0.0f), glm::vec3(1.0, 0.0, 0.0));
        

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
            cv::imwrite(filename.str(), colorImage);

            std::cout << "Wrote segmentation of: " << scene_name << std::endl;
        }

        // show image in window
        glfwSwapBuffers(m_window);
    
    }
    
}