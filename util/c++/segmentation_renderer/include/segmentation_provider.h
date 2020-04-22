#pragma once

#include "mesh.h"

#include "json.hpp"
// for convenience
using json = nlohmann::json;

#include <map>
#include <glm/glm.hpp>

// label_to_color or object_id_to_color??

/*
std::map <std::string, std::vector<int>> label_to_color { 
  { "lamp", {255, 0, 255} }, 
  { "pillow", {255, 255, 0} } 
};

void random_rgb(int rgb[]);
void random_rgb(int rgb[])
{
  int i;
  for(i=0;i<3;i++)
  {
    rgb[i]=rand()%256;
  }
}
*/

class Segmentation_Provider {

    public:
        Segmentation_Provider(string const &semseg_path, string const &vseg_path);
        ~Segmentation_Provider();
        void change_colors(Mesh &mesh);

    private:
        int n_objects;
        int n_vertices;
        json vseg;
        json semseg;
        std::vector<int> vertex_to_segment;
        std::map<int, int> segment_to_object_id;;
        std::map<int, glm::vec3> object_id_to_color;

        void random_rgb(float rgb[]){
            int i;
            for(i=0;i<3;i++){
                // this produces a random float number between 0 and 1 inclusively
                rgb[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            }
        }
};