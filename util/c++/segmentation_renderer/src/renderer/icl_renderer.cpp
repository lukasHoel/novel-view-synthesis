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

        // trans = glm::translate(trans, glm::vec3(-2.75f, 0.0f, 0.0f));
    trans = glm::scale(trans, glm::vec3(-1, -1, -1));
        // trans = glm::scale(trans, glm::vec3(-1, 1, 1));
    // trans = glm::rotate(trans, glm::radians(180.0f), glm::vec3(0.0, 1.0, 0.0));
        // trans = glm::rotate(trans, glm::radians(270.0f), glm::vec3(1.0, 0.0, 0.0));




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
        // trans = glm::rotate(trans, glm::radians(180.0f), glm::vec3(0.0, 0.0, 1.0));
        // trans = glm::scale(trans, glm::vec3(1, 1, -1));

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

        float min = -1;
        float max = -1;

        for(int i=0; i<colorImage.rows; i++){
            for(int j=0; j<colorImage.cols; j++){
                
                cv::Vec3b &rgb = colorImage.at<cv::Vec3b>(i,j);
                // std::cout << static_cast<unsigned>(rgb.val[0]) << static_cast<unsigned>(rgb.val[1]) << static_cast<unsigned>(rgb.val[2]) << std::endl;

                if(min == -1 || rgb.val[0] < min){
                    min = rgb.val[0];
                }
                if(max == -1 || rgb.val[0] > max){
                    max = rgb.val[0];
                }
            }
        }

        for(int i=0; i<colorImage.rows; i++){
            for(int j=0; j<colorImage.cols; j++){
                cv::Vec3b &rgb = colorImage.at<cv::Vec3b>(i,j);
                float OldRange = max - min;
                float NewRange = 255;

                rgb.val[0] = (((rgb.val[0] - min) * NewRange) / OldRange) + 0;
                rgb.val[1] = (((rgb.val[1] - min) * NewRange) / OldRange) + 0;
                rgb.val[2] = (((rgb.val[2] - min) * NewRange) / OldRange) + 0;
            }
        }

        cv::imshow("color image", colorImage); 
        cv::waitKey(0);



        // read depth image into openCV matrix
        cv::Mat depthImage;
        readDepth(depthImage);
        // cv::imshow("depth image", depthImage); 
        // cv::waitKey(0);

        GLint viewport[4];
        GLdouble modelview[16];
        GLdouble proj[16];
        
        // glGetDoublev( GL_MODELVIEW_MATRIX, modelview );
        // glGetDoublev( GL_PROJECTION_MATRIX, proj );
        // glGetIntegerv( GL_VIEWPORT, viewport );

        min = -1;
        max = -1;

        for(int i=0; i<depthImage.rows; i++){
            for(int j=0; j<depthImage.cols; j++){
                GLdouble posX, posY, posZ; // outputs for gluUnProject
                
                const float z = depthImage.at<float>(i,j);
                const float zn = (2 * z - 1);
                const float ze = (2 * kFarPlane * kNearPlane) / (kFarPlane + kNearPlane + zn*(kNearPlane - kFarPlane));
                
                depthImage.at<float>(i,j) = ze * 255 / kFarPlane;

                // std::cout << "BEFORE: " << depthImage.at<float>(i,j) << std::endl;
                // gluUnProject(1.0 * j / depthImage.cols, 1.0 * i / depthImage.rows, z, modelview, proj, viewport, &posX, &posY, &posZ);
                // depthImage.at<float>(i,j) = posZ;

                if(min == -1 || depthImage.at<float>(i,j) < min){
                    min = depthImage.at<float>(i,j);
                }
                if(max == -1 || depthImage.at<float>(i,j) > max){
                    max = depthImage.at<float>(i,j);
                }
                // std::cout << "AFTER: " << depthImage.at<float>(i,j) << std::endl;
            }
        }

        for(int i=0; i<depthImage.rows; i++){
            for(int j=0; j<depthImage.cols; j++){
                float z = depthImage.at<float>(i,j);
                float OldRange = max - min;
                float NewRange = 255;

                depthImage.at<float>(i,j) = (((z - min) * NewRange) / OldRange) + 0;
            }
        }

        // cv::imshow("depth image", depthImage); 
        // cv::waitKey(0);

        // save matrix as file
        if (save_path != "") {
            std::stringstream filename;
            char scene_name[30];
            sprintf(scene_name, "scene_%02d_%04d", ip.getSceneNr(), i);
            filename << save_path << "/" << scene_name << ".png";
            // cv::imwrite(filename.str(), colorImage);

            std::stringstream depth_filename;
            depth_filename << save_path << "/" << scene_name << ".depth.png";
            // cv::imwrite(depth_filename.str(), depthImage);

            std::cout << "Wrote segmentation of: " << scene_name << std::endl;
        }

        // show image in window
        glfwSwapBuffers(m_window);
    
    }
    
}