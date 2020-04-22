#include "segmentation_provider.h"

Segmentation_Provider::Segmentation_Provider(string const &semseg_path,
                                             string const &vseg_path) {
    // read semseg file
    std::ifstream semseg_file(semseg_path);
    semseg_file >> semseg;

    // read vseg file
    std::ifstream vseg_file(vseg_path);
    vseg_file >> vseg;

    // Number of Vertices as defined in the vseg file
    n_vertices = vseg["segIndices"].size();

    // Number of objects as defined in the semseg file
    n_objects = semseg["segGroups"].size();
    
    // Vertex to Segment map
    for (auto& segment : vseg["segIndices"]) {
        // the json file contains the segments listed in order of the vertices, so first entry is segment of first vertex
        vertex_to_segment.push_back(segment);
    }

    // Segment to Object map
    for (auto& group : semseg["segGroups"]) {
        int id = group["id"];
        // read all segments for this group
        for (auto& segment : group["segments"]) {
            segment_to_object_id[segment] = id;
        }
    }

    // Object to Color Map
    for (auto i=0; i<n_objects; i++) {
        float rgb[3];
        random_rgb(rgb);
        object_id_to_color[i] = glm::vec3(rgb[0], rgb[1], rgb[2]);
    }
}

Segmentation_Provider::~Segmentation_Provider() = default;

void Segmentation_Provider::change_colors(Mesh &mesh){
    for(auto i=0; i<mesh.vertices.size(); i++){

        // look up the new color for this vertex
        int segment = vertex_to_segment[i];
        int object_id = segment_to_object_id[segment];
        glm::vec3 rgb = object_id_to_color[object_id];

        // assign new color
        mesh.vertices[i].Color = rgb;
    }

    // update information on GPU for OpenGL to render with new values
    mesh.updateData();
}