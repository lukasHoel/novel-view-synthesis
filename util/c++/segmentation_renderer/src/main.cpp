#include <iostream>

#include "renderer.h"
#include "segmentation_provider.h"
#include "mp.h"
#include "mesh_transformer.h"
#include "util.h"

#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include <glm/glm.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>

#include <glm/gtc/type_ptr.hpp>

int main(int argc, char** argv){
    if(argc != 3){
        std::cout << "Usage: " << argv[0] << " path/to/Matterport3D/data/v1/scans <scanID>" << std::endl;
        return EXIT_FAILURE;
    }

    string path(argv[1]);
    string scanID(argv[2]);

    string pathToHouseFile = path + "/" + scanID + "/house_segmentations/" + scanID + "/house_segmentations/" + scanID + ".house";
    MP_Parser mp(pathToHouseFile.c_str());
    std::cout << "parsed .house file" << std::endl;

    for(MPImage* image : mp.regions[0]->panoramas[0]->images){
        std::cout << "Image in region 0: " << image->color_filename << std::endl;
        std::cout << "Intrinsics: " << glm::to_string(glm::make_mat3(image->intrinsics)) << std::endl;
        std::cout << "Extrinsics: " << glm::to_string(glm::make_mat4(image->extrinsics)) << std::endl;

        glm::mat3 intr = glm::make_mat3(image->intrinsics);

        std::cout << "Perspective: " << glm::to_string(camera_utils::perspective(intr, image->width, image->height, 0.001, 10)) << std::endl;
    }

    try{
        string regionPath = path + "/" + scanID + "/region_segmentations/" + scanID + "/region_segmentations";

        regionPath += "/region0."; // TODO instead loop over all regions!

        Renderer renderer(regionPath + "ply");

        std::cout << "Renderer initialized" << std::endl;

        Segmentation_Provider sp(regionPath + "semseg.json", regionPath + "vsegs.json");
        for(auto& mesh : renderer.m_model->meshes){
            sp.change_colors(mesh);
        }

        std::cout << "Updated Mesh Colors according to object instance segmentation" << std::endl;

        for(auto& mesh : renderer.m_model->meshes){
            Mesh_Transformer transform(mesh, sp);
            transform.splitMeshAtObject(16);
            glm::mat4 t = { 1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0,
                            0.5, 0.5, 0.5, 1 };
            transform.moveVerticesOfObject(16, t);
        }

        std::cout << "Splitted and transformed mesh: bed translated by (0.5, 0.5, 0.5)" << std::endl;

        renderer.renderInteractive();

    } catch(const exception& e){
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    

    return EXIT_SUCCESS;
}