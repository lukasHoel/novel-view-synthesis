import bpy
import bmesh
import os
import struct
import math
import mathutils

HEADER = \
"ply\n" + \
"format binary_little_endian 1.0\n" + \
"comment Written with hapPLY (https://github.com/nmwsharp/happly)\n" + \
"element vertex {vcount}\n" + \
"property float x\n" + \
"property float y\n" + \
"property float z\n" + \
"property uchar red\n" + \
"property uchar green\n" + \
"property uchar blue\n" + \
"element face {fcount}\n" + \
"property list uchar int vertex_indices\n" + \
"property int object_id\n" + \
"end_header\n"

vertex_to_rgb = []
face_to_objID = []
objID_to_face = {}

def importer(path, vertex_to_rgb, face_to_objID, objID_to_face):
    vertex_to_rgb.clear()
    face_to_objID.clear()
    objID_to_face.clear()
    # Read PLY file
    data = None
    with open(path, "rb") as file:
        data = file.read()

    is_binary = data.find(b"ascii", 0, 20) == -1
    
    # Binary PLY
    if is_binary:
        print("Parsing binary file...")
        
        # Count number of lines for header (may vary)
        HEADER_SIZE = 0
        VERTEX_COUNT = None
        FACE_COUNT = None
        
        line = b""
        start = 0
        max = len(data)
        while line.lower() != "end_header":
            end = data.find(b"\n", start, max)
            line = data[start:end].decode("ascii")
            if "element vertex" in line:
                VERTEX_COUNT = int(line.split(" ")[-1])
            elif "element face" in line:
                FACE_COUNT = int(line.split(" ")[-1])
                
            HEADER_SIZE += end - start + 1
            start = end + 1
 
        print("HEADER_SIZE (in bytes):", HEADER_SIZE)
        print("VERTEX_COUNT:", VERTEX_COUNT)
        print("FACE_COUNT:", FACE_COUNT)
        
        bytes_per_v = 4 + 4 + 4 + 1 + 1 + 1
        bytes_per_f = 1 + 4 + 4 + 4 + 4
        v_start = HEADER_SIZE
        v_end = v_start + bytes_per_v * VERTEX_COUNT
        
        v_bytes = data[v_start:v_end]
        f_bytes = data[v_end:]

        for i in range(0, len(v_bytes), bytes_per_v):
            vertex_to_rgb.append(list(struct.unpack("BBB", v_bytes[i+12:i+15]))) # B: uchar
        
        face_id = 0
        for i in range(0, len(f_bytes), bytes_per_f):
            objID, = struct.unpack("<i", f_bytes[i+13:i+17]) # <: little-endian, i: int
            face_to_objID.append(objID)
            
            if objID in objID_to_face:
                objID_to_face[objID].append(face_id)
            else:
                objID_to_face[objID] = [face_id]
                
            face_id += 1
        
    # ASCII PLY
    else:
        print("Parsing ASCII file...")
        data = data.decode("ascii").splitlines()

        # Count number of lines for header (may vary)
        HEADER_SIZE = 1
        VERTEX_COUNT = None
        FACE_COUNT = None
        for line in data:
            if "element vertex" in line:
                VERTEX_COUNT = int(line.split(" ")[-1])
            elif "element face" in line:
                FACE_COUNT = int(line.split(" ")[-1])
            elif "end_header" in line.lower():
                break
                
            HEADER_SIZE += 1

        print("HEADER_SIZE:", HEADER_SIZE)
        print("VERTEX_COUNT:", VERTEX_COUNT)
        print("FACE_COUNT:", FACE_COUNT)
        
        # Extract per vertex color and per face object ID information
        v_start = HEADER_SIZE
        v_end = v_start + VERTEX_COUNT
        vertex_info = data[v_start:v_end]
        face_info = data[v_end:]

        for i, line in enumerate(vertex_info):
            rgb = list(map(lambda x: int(x), line.split(" ")[-3:]))
            vertex_to_rgb.append(rgb)
            
        for i, line in enumerate(face_info):
            objID = int(line.split(" ")[-1])
            face_to_objID.append(objID)
            
            if objID in objID_to_face:
                objID_to_face[objID].append(i)
            else:
                objID_to_face[objID] = [i]
    print("Importing finished.")
        

def exporter(path, mesh, vertex_to_rgb, face_to_objID):
    global HEADER
    verts = mesh.verts
    faces = mesh.faces
    with open(path, "wb+") as ply:
        header = HEADER.format(vcount=len(verts), fcount=len(faces))
        ply.write(header.encode("ascii"))

        for i, v in enumerate(mesh.verts):
            buf = struct.pack('fffBBB', *v.co, *vertex_to_rgb[i]) # f: float, B: uchar
            ply.write(buf)

        for i, f in enumerate(mesh.faces):
            v_idxs = list(map(lambda x: x.index, f.verts))
            buf = struct.pack('<Biiii', 3, *v_idxs, face_to_objID[i]) # <: little-endian B: uchar, i: int
            ply.write(buf)
            
def get_objID_of_selection(mesh, face_to_objID):
    # Face info for selection
    for f in mesh.faces:
        if f.select:
            face_idx = f.index  # Blender idx (zero-indexed) -> PLY file line idx (zero-indexed)
            objID = face_to_objID[face_idx]
            print("Face ID: {} belongs has the object ID: {}".format(face_idx, objID))
                
def select_faces_of_obj(objID, objID_to_face, mesh, obj_data, sel_extend=False):
    # Deselect all previous if extended selection is not desired
    if not sel_extend:
        for f in mesh.faces:
            f.select = False

    # Select faces of a specific object
    idxs = objID_to_face[objID]
    for i in idxs:
        mesh.faces[i].select = True

    bmesh.update_edit_mesh(obj_data, True)
    
def get_objIDs_of_vertex(v, face_to_objID):
    """Get all objID's bound to a specific vertex v"""
    adj_faces = v.link_faces
    obj_ids = list(map(lambda x: face_to_objID[x.index], adj_faces))
    return obj_ids
    
def cut_object(objID, objID_to_face, face_to_objID, mesh, obj_data):
    """Cuts the faces of objID from its neighbours. Faces whose one or more edges are removed are deleted."""
    idxs = objID_to_face[objID]
    del_faces = set()
    del_edges = set()
    # For each face f of object
    for i in idxs:

       # For each vertex v of the face
       for v in mesh.faces[i].verts:

           # For each edge e originating from v
           for e in v.link_edges:
               this = v.index
               v1, v2 = e.verts
               
               # Determine other endpoint of the edge
               that = v1
               if this == that.index:
                   that = v2
               
               # Determine the object IDs bound to the other endpoint
               that_objIDs = get_objIDs_of_vertex(that, face_to_objID)
               # Remove edge if other endpoint doesn't belong to same object
               if objID not in that_objIDs:
                   # Required to update face-objID mapping
                   for f in that.link_faces:
                     if e in f.edges:
                        del_faces.add(f.index)

                   del_edges.add(e)

    # Update face-objID mappings
    objID_to_face.clear() # Needs to be rebuilt due to face index shift 
    for face_id in sorted(del_faces, reverse=True):
        del face_to_objID[face_id]

    for face_id, objID in enumerate(face_to_objID):
        if objID in objID_to_face:
            objID_to_face[objID].append(face_id)
        else:
            objID_to_face[objID] = [face_id]


    bmesh.ops.delete(mesh, geom=list(del_edges), context="EDGES_FACES")
    bmesh.update_edit_mesh(obj_data, True)
    
def get_RT_matrix(angles, t, in_degrees=False):
    """Specify rotation angles (as (rx, ry, rz) in radians by default) and translation parameters (as (x,y,z)), return equivalent 3x4 transformation matrix"""
    if in_degrees:
        angles = [math.radians(angle) for angle in angles]
    # Get 4x4 rotation matrix
    euler = mathutils.Euler(angles, 'XYZ')
    R = euler.to_matrix().to_4x4()
    # Generate 3x4 RT
    RT = mathutils.Matrix(R[0:3])
    # Embed translation
    RT.col[3] = mathutils.Vector(t)
    # Return overall RT matrix as 3x4
    return RT

def selection_center(mesh):
    """Calculate the center of selected faces. Required for pivot point transform during rotation."""
    selected_verts = list(filter(lambda v: v.select, mesh.verts))
    coords = [v.co for v in selected_verts]
    center = mathutils.Vector((0, 0, 0))
    for c in coords:
      center += c
    center /= len(coords)
    return center

def translate_selection(t):
    """Translate with t (x,y,z) using blender ops directly, return equivalent 4x4 translation matrix"""
    bpy.ops.transform.translate(value=t, 
                                orient_type='GLOBAL', 
                                orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                                orient_matrix_type='GLOBAL', 
                                constraint_axis=(False, True, False), 
                                mirror=True, 
                                use_proportional_edit=False, 
                                proportional_edit_falloff='SMOOTH', 
                                proportional_size=1, 
                                use_proportional_connected=False, 
                                use_proportional_projected=False, 
                                release_confirm=True)
    return mathutils.Matrix.Translation(t)

def rotate_selection(mesh, angle, axis, in_degrees=False):
    """Rotate using blender ops directly (as float, in radians by default) around axis 'X', 'Y' or 'Z', return equivalent 4x4 rotation matrix."""
    axis = axis.upper()
    if axis not in ['X', 'Y', 'Z']:
        return None
    if in_degrees:
        angle = math.radians(angle)
    bpy.ops.transform.rotate(value=angle, 
                             orient_axis=axis, 
                             orient_type='GLOBAL', 
                             orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                             orient_matrix_type='GLOBAL', 
                             constraint_axis=(False, True, False), 
                             mirror=True, 
                             use_proportional_edit=False, 
                             proportional_edit_falloff='SMOOTH', 
                             proportional_size=1, 
                             use_proportional_connected=False,
                             use_proportional_projected=False, 
                             release_confirm=True)
    pivot = selection_center(mesh)
    origin_map = mathutils.Matrix.Translation(-pivot)
    rotation = mathutils.Matrix.Rotation(angle, 4, axis)
    inv_origin_map = mathutils.Matrix.Translation(pivot)
    return inv_origin_map @ rotation @ origin_map
                             
def transform_selection(RT, obj_data):
    """
    Apply transformation matrix on selected vertices.
    Note that blender ops not used. This method can be used for testing the transformation matrix generated with composite transformations.
    """
    if len(RT) == 4:
        RT = mathutils.Matrix(RT[0:3])
    selected_verts = list(filter(lambda v: v.select, mesh.verts))
    for v in selected_verts:
        v.co = RT @ mathutils.Vector((*v.co, 1))

    bmesh.update_edit_mesh(obj_data, True)