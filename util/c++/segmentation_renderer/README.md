# Segmentation Renderer
The segmentation renderer allows to render instance segmentation masks out of the original Matterport3D camera images. You can also choose to move certain objects of the mesh arouny by an arbitrary transformation matrix. 

## Prerequisites
- at least c++11 capable compiler (gcc, ...)
   - Check if installed: gcc --version
- cmake
   - sudo apt-get install cmake

## Setup (Dependencies for this program)

    sudo apt-get install assimp-utils libassimp-dev
    sudo apt-get install libopencv-dev
    sudo apt-get install libglm-dev
    sudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev
    sudo apt-get install libglfw3-dev libglfw3
    sudo apt-get install libglew-dev

## Build the Program
    mkdir build
    cd build
    cmake ..
    make

## Run the Program
    ./segmentation_renderer /path/to/Matterport3D/data/v1/scans <scanID>

    ./segmentation_renderer ~/datasets/Matterport3D/data/v1/scans 17DRP5sb8fyD
