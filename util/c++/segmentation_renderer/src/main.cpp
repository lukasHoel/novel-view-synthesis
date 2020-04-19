#include <iostream>

#include "renderer.h"

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
    if(argc != 2){
        std::cout << "Usage: " << argv[0] << " path/to/mesh.obj" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Working Directory: " << GetCurrentWorkingDir() << std::endl;

    try{
        Renderer renderer(argv[1]);

        std::cout << "Renderer initialized" << std::endl;

        renderer.renderInteractive();

    } catch(const exception& e){
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}