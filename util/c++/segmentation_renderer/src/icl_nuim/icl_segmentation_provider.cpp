#include "icl_segmentation_provider.h"

ICL_Segmentation_Provider::ICL_Segmentation_Provider(string const &object_to_color_path) {
    // read object_to_color file
    std::ifstream f(object_to_color_path);
    
    if(! f.is_open()){
        throw std::runtime_error("Could not read object_to_color file from: " + object_to_color_path);
    }

    std::string name;
    float r, g, b;

    while(f >> name >> r >> g >> b){
        object_name_to_color[name] = glm::vec3(r, g, b);
    }
}

ICL_Segmentation_Provider::~ICL_Segmentation_Provider() = default;

void ICL_Segmentation_Provider::change_colors(Mesh &mesh){
    for(auto i=0; i<mesh.vertices.size(); i++){

        // look up the new color for this vertex
        glm::vec3 rgb = object_name_to_color[mesh.vertices[i].Name];

        // assign new color
        mesh.vertices[i].Color = rgb;
    }

    // update information on GPU for OpenGL to render with new values
    mesh.updateData();
}