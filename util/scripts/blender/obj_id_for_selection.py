import bpy
import bmesh
import os

# Args for the program
ASCII_PATH = "path/to/ascii.ply"

ascii_data = None
with open(ASCII_PATH) as file:
    ascii_data = file.readlines()

HEADER_SIZE = 1
for line in ascii_data:
    if "end_header" in line or "END_HEADER" in line:
        break
    HEADER_SIZE += 1

obj = bpy.context.object             # Get mesh in the scene
bpy.ops.object.mode_set(mode='EDIT') # Put blender into edit mode
bm = bmesh.from_edit_mesh(obj.data)  # Access faces and vertices of obj

VERTEX_COUNT = len(bm.verts)

print("HEADER_SIZE:", HEADER_SIZE)
print("VERTEX_COUNT:", VERTEX_COUNT)
    
# Face info for selection
for f in bm.faces:
    if f.select:
        ply_index = HEADER_SIZE + VERTEX_COUNT + f.index  # Blender idx (zero-indexed) -> PLY file line idx (zero-indexed)
        obj_id = int(ascii_data[ply_index].split(" ")[-1])
        print("Face ID: {} belongs has the object ID: {}".format(ply_index, obj_id))
        
# Vertex info for selection
#    for v in bm.verts:
#        if v.select:
#            print("Vertex ID:", v.index)
#            print("Vertex Coords:", v.co)