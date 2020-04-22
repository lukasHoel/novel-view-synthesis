#include <iostream>

#include "renderer.h"

#include "segmentation_provider.h"

#include <stdio.h>
#include <unistd.h>
#define GetCurrentDir getcwd

std::string GetCurrentWorkingDir( void ) {
  char buff[FILENAME_MAX];
  GetCurrentDir( buff, FILENAME_MAX );
  std::string current_working_dir(buff);
  return current_working_dir;
}

int main(int argc, char** argv){
    if(argc != 4 && argc != 2){
        std::cout << "Usage: " << argv[0] << " path/to/mesh.obj [path/to/semseg.json path/to/vseg.json]" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Working Directory: " << GetCurrentWorkingDir() << std::endl;
    
    try{
        Renderer renderer(argv[1]);

        std::cout << "Renderer initialized" << std::endl;

        if(argc == 4){
            Segmentation_Provider sp(argv[2], argv[3]);
            for(auto& mesh : renderer.m_model->meshes){
                sp.change_colors(mesh);
            }

            std::cout << "Updated Mesh Colors according to object instance segmentation" << std::endl;
        }

        renderer.renderInteractive();

    } catch(const exception& e){
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    

    return EXIT_SUCCESS;
}