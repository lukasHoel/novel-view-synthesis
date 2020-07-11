from sys import argv
from utils import *

# NOTE: Import this file from blender python console

# Args for the program
INPUT_PATH = "/path/to/_semantic.ply"
OUTPUT_PATH = "/path/to/new_semantic.ply"

# Retrieve data from blender
obj = bpy.context.object              # Get mesh in the scene
bpy.ops.object.mode_set(mode='EDIT')  # Put blender into edit mode
obj_data = obj.data
mesh = bmesh.from_edit_mesh(obj.data) # Access faces and vertices of obj

# If index error happens due to outdated internal index table, use the followings:
# mesh.verts.ensure_lookup_table()
# mesh.faces.ensure_lookup_table()
# mesh.edges.ensure_lookup_table()

VERTEX_COUNT = len(mesh.verts)
FACE_COUNT = len(mesh.faces)

# File has to be parsed again manually in the beginning because blender doesn't read vertex colors and object IDs. 
importer(INPUT_PATH, vertex_to_rgb, face_to_objID, objID_to_face)

#####################################
# Available operations and examples:#
#####################################

# Select faces from UI first then (make sure to use face selection, i.e. not edge, not vertex):
# get_objID_of_selection(mesh, face_to_objID)

# select_faces_of_obj(17, objID_to_face, mesh, obj_data, sel_extend=False)

# cut_object(17, objID_to_face, face_to_objID, mesh, obj_data)

# Transformations via console:
# R = rotate_selection(mesh, 45, 'X', in_degrees=True)
# T = translate_selection((1, 0, -1))

# Alternatively, transformations via UI: Move (bpy.ops.transform.translate(value=(x,y,z))) & Rotate (bpy.ops.transform.rotate(value=radians, orient_axis=axis,..)
# Check the logs and by passing "value" and "axis" fields into respective args in get_RT_matrix, one can get overall RT matrix:
# RT = get_RT_matrix((30, 0, 0), (1, 0, -1), in_degrees=True), rotation vector (30, 0, 0) implies rotation is around "X" axis with 30 degrees

# Used internally to translate center of selected faces to origin during rotation. However, it can be useful in debugging as well.
# center = selection_center(mesh)

# Perform multiple translate_selection rotate_selection calls and multiply matrices in correct order. 
# Then, call this function with inverted composite transformation. It's a good debug tool.
# transform_selection(RT, obj_data)

# Finally, export modified mesh with:
# exporter(OUTPUT_PATH, mesh, vertex_to_rgb, face_to_objID)

