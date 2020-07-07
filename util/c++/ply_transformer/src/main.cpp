#include <iostream>

#include "happly.h"

#include <map>
#include <algorithm>

int main(int argc, char** argv){

    if(argc != 3){
        std::cerr << "Usage: " << argv[0] << " path/to/binary.ply output/path/binary_modified.ply" << std::endl;
        return EXIT_FAILURE;
    }

    /********************
     * READ MESH
     *******************/

    // Construct a data object by reading from file
    happly::PLYData plyIn(argv[1]);

    // Get mesh-style data from the object
    std::vector<std::array<float, 3>> vertices = plyIn.getVertexPositions<float>();
    std::vector<std::vector<int>> faces = plyIn.getFaceIndices<int>();
    std::vector<std::array<unsigned char, 3>> colors = plyIn.getVertexColors();

    // Get object_id
    std::vector<int> object_ids = plyIn.getElement("face").getProperty<int>("object_id");

    /********************
     * MODIFY MESH
     *******************/

    // OPTION A: brute-force modify some vertices
    int start = 0;
    int end = 10000;

    // for(int i=start; i<end; i++){
    //     vertices[i][0] += 1;
    //     vertices[i][1] += 1;
    //     vertices[i][2] += 1;
    // }

    // OPTION B: modify all vertices that belong to faces with object_id <foo>
    std::vector<int> object_ids_to_be_moved = {1, 3, 7, 10, 15, 19, 23, 26, 42, 47, 11}; // randomly chosen list of object_ids: I have no idea what these ids actually reference
    std::map<int, bool> already_modified_vertices;
    for(int i=0; i<faces.size(); i++){

        // if the object id for this face is in the above list
        if(std::find(object_ids_to_be_moved.begin(), object_ids_to_be_moved.end(), object_ids[i]) != object_ids_to_be_moved.end()){
            // change all vertices of this face, if they were not already changed once
            for(const auto& index : faces[i]){
                // if the vertex was not already moved while iterating over a previous face
                if(already_modified_vertices.find(index) == already_modified_vertices.end()){
                    // std::cout << "move vertex " << index << std::endl;
                    // change vertex
                    vertices[index][0] += 1;
                    vertices[index][1] += 1;
                    vertices[index][2] += 1;
                    // set in map
                    already_modified_vertices[index] = true;
                }
            }
        }
    }

    /********************
     * WRITE MESH
     *******************/

    // Create an empty object
    happly::PLYData plyOut;

    // Add mesh data (elements are created automatically)
    plyOut.addVertexPositions(vertices);
    plyOut.addVertexColors(colors);
    plyOut.addFaceIndices(faces);

    // Add object_id
    plyOut.getElement("face").addProperty<int>("object_id", object_ids);

    // Write the object to file
    plyOut.write(argv[2], happly::DataFormat::Binary);

    return EXIT_SUCCESS;
}