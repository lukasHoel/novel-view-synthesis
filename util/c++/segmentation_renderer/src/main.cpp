#include <iostream>

#include "renderer.h"
#include "segmentation_provider.h"
#include "mp.h"
#include "mesh_transformer.h"

#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include <glm/glm.hpp>

/*
        - TODO: Write a main.cpp pipeline which loops over all images (selected images --> loop over folder of images) for a specific region
            --> Color according to segmentation + transform object from input
            --> Render from specific camera pose/intrinsic for this view
            --> Save as image in output folder
        - TODO (optional): Before the main.cpp pipeline starts, we show the region in an interactive renderer.
            --> Allow the user to somehow interactively choose which object to move and how to move it
            --> From this selection, extract a transformation matrix and use that as an input for the pipeline
            --> (optional 2): Let the user create a trajectory (multiple transformation matrices) and use each of them
*/

int main(int argc, char** argv){
    if(argc != 3){
        std::cout << "Usage: " << argv[0] << " path/to/Matterport3D/data/v1/scans <scanID>" << std::endl;
        return EXIT_FAILURE;
    }

    string path(argv[1]);
    string scanID(argv[2]);
    string outdir = path + "/" + scanID + "/image_segmentations"; // TODO mkdir this

    string pathToHouseFile = path + "/" + scanID + "/house_segmentations/" + scanID + "/house_segmentations/" + scanID + ".house";
    MP_Parser mp(pathToHouseFile.c_str());
    std::cout << "parsed .house file" << std::endl;

    try{
        string regionPath = path + "/" + scanID + "/region_segmentations/" + scanID + "/region_segmentations";

        regionPath += "/region0."; // TODO instead loop over all regions!

        Renderer renderer(regionPath + "ply", mp, 0);

        std::cout << "Renderer initialized" << std::endl;

        Segmentation_Provider sp(regionPath + "semseg.json", regionPath + "vsegs.json");
        // for(auto& mesh : renderer.m_model->meshes){
        //     sp.change_colors(mesh);
        // }
        // renderer.renderInteractive();

        // std::cout << "Updated Mesh Colors according to object instance segmentation" << std::endl;

        // // render original (before move)
        // renderer.renderImages(outdir + "/original");

        for(auto& mesh : renderer.m_model->meshes){
            Mesh_Transformer transform(mesh, sp);
            transform.splitMeshAtObject(11);
            glm::mat4 t = { 1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0,
                            0, -0.5, 0.5, 1 };
            transform.moveVerticesOfObject(11, t);
        }
        renderer.renderInteractive();

        std::cout << "Splitted and transformed mesh: pillow with id 11 translated by (0, -0.5, 0.5)" << std::endl;
        string save_path("x.ply");
        renderer.m_model->save(save_path);

        // render moved
        // renderer.renderImages(outdir + "/moved");

        // std::cout << "Render images completed" << std::endl;

    } catch(const exception& e){
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    

    return EXIT_SUCCESS;
}