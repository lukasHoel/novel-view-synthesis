import bpy
import bmesh
import os

# Args for the program
ASCII_PATH = "path/to/ascii.ply"
SELECT_OBJ = 1

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
FACE_COUNT = len(bm.faces)

print("HEADER_SIZE:", HEADER_SIZE)
print("VERTEX_COUNT:", VERTEX_COUNT)
print("FACE_COUNT:", FACE_COUNT)

# Jump to faces (Zero-indexed)
face_start = HEADER_SIZE + VERTEX_COUNT
faces = ascii_data[face_start:]
# Extract object IDs
obj_ids = list(map(lambda x: x.split(" ")[-1], faces))
find = lambda query: [i for i, obj_id in enumerate(obj_ids) if query == int(obj_id)]

# Deselect all previous (This part should be disabled when extending the selection is desired)
for f in bm.faces:
    f.select = False

# Select faces of a specific object
idxs = find(SELECT_OBJ)
for i in idxs:
    bm.faces[i].select = True

bmesh.update_edit_mesh(obj.data, True)