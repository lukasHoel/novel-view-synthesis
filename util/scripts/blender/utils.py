import bpy
import bmesh
import json
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

SEG_COLORS = [[0.12156863, 0.46666667, 0.70588235],
              [0.68235294, 0.78039216, 0.90980392],
              [1.        , 0.49803922, 0.05490196],
              [1.        , 0.73333333, 0.47058824],
              [0.17254902, 0.62745098, 0.17254902],
              [0.59607843, 0.8745098 , 0.54117647],
              [0.83921569, 0.15294118, 0.15686275],
              [1.        , 0.59607843, 0.58823529],
              [0.58039216, 0.40392157, 0.74117647],
              [0.77254902, 0.69019608, 0.83529412],
              [0.54901961, 0.3372549 , 0.29411765],
              [0.76862745, 0.61176471, 0.58039216],
              [0.89019608, 0.46666667, 0.76078431],
              [0.96862745, 0.71372549, 0.82352941],
              [0.49803922, 0.49803922, 0.49803922],
              [0.78039216, 0.78039216, 0.78039216],
              [0.7372549 , 0.74117647, 0.13333333],
              [0.85882353, 0.85882353, 0.55294118],
              [0.09019608, 0.74509804, 0.81176471],
              [0.61960784, 0.85490196, 0.89803922],
              [0.22352941, 0.23137255, 0.4745098 ],
              [0.32156863, 0.32941176, 0.63921569],
              [0.41960784, 0.43137255, 0.81176471],
              [0.61176471, 0.61960784, 0.87058824],
              [0.38823529, 0.4745098 , 0.22352941],
              [0.54901961, 0.63529412, 0.32156863],
              [0.70980392, 0.81176471, 0.41960784],
              [0.80784314, 0.85882353, 0.61176471],
              [0.54901961, 0.42745098, 0.19215686],
              [0.74117647, 0.61960784, 0.22352941],
              [0.90588235, 0.72941176, 0.32156863],
              [0.90588235, 0.79607843, 0.58039216],
              [0.51764706, 0.23529412, 0.22352941],
              [0.67843137, 0.28627451, 0.29019608],
              [0.83921569, 0.38039216, 0.41960784],
              [0.90588235, 0.58823529, 0.61176471],
              [0.48235294, 0.25490196, 0.45098039],
              [0.64705882, 0.31764706, 0.58039216],
              [0.80784314, 0.42745098, 0.74117647],
              [0.87058824, 0.61960784, 0.83921569]]

#              Global variables used frequently in functions
obj = obj_data = mesh = VERTEX_COUNT = FACE_COUNT = None

vertex_to_rgb = []
face_to_objID = []
objID_to_face = {}

# List of transformation matrices stored when translate_selection, rotate_selection or transform_selection are called.
# Later transformation matrices can be composed with compose_transforms function
transforms = []

from_blender_to_habitat = mathutils.Matrix([[1,  0, 0, 0], 
                                            [0,  0, 1, 0], 
                                            [0, -1, 0, 0],
                                            [0,  0, 0, 1]])
from_habitat_to_blender = from_blender_to_habitat.inverted()

def blender_init():
    """
    Sets global variables that are required for scene manipulation. 
    Undoing an action may delete existing global variables. If that's the case, call this method again.
    """
    global obj, obj_data, mesh, VERTEX_COUNT, FACE_COUNT
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

def importer(path):
    global vertex_to_rgb, face_to_objID, objID_to_face
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
        

def exporter(path):
    global HEADER, mesh, vertex_to_rgb, face_to_objID

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
            
def get_objID_of_selection():
    global mesh, face_to_objID

    # Face info for selection
    for f in mesh.faces:
        if f.select:
            face_idx = f.index  # Blender idx (zero-indexed) -> PLY file line idx (zero-indexed)
            objID = face_to_objID[face_idx]
            print("Face ID: {} belongs has the object ID: {}".format(face_idx, objID))
                
def select_faces_of_obj(objID, sel_extend=False):
    global objID_to_face, mesh, obj_data

    # Deselect all previous if extended selection is not desired
    if not sel_extend:
        for f in mesh.faces:
            f.select = False

    # Select faces of a specific object
    idxs = objID_to_face[objID]
    for i in idxs:
        mesh.faces[i].select = True

    bmesh.update_edit_mesh(obj_data, True)
    
def get_objIDs_of_vertex(v):
    global face_to_objID

    """Get all objID's bound to a specific vertex v"""
    adj_faces = v.link_faces
    obj_ids = list(map(lambda x: face_to_objID[x.index], adj_faces))
    return obj_ids
    
def cut_object(objID):
    global objID_to_face, face_to_objID, mesh, obj_data

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
               that_objIDs = get_objIDs_of_vertex(that)
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

def selection_center():
    global mesh

    """Calculate the center of selected faces. Required for pivot point transform during rotation."""
    selected_verts = list(filter(lambda v: v.select, mesh.verts))
    coords = [v.co for v in selected_verts]
    center = mathutils.Vector((0, 0, 0))
    for c in coords:
      center += c
    center /= len(coords)
    return center

def translate_selection(t, store=True):
    global transforms

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
    t_matrix = mathutils.Matrix.Translation(t)
    if store:
        transforms.append(t_matrix)
    return t_matrix

def rotate_selection(angle, axis, store=True, in_degrees=False):
    global mesh, transforms

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
    pivot = selection_center()
    origin_map = mathutils.Matrix.Translation(-pivot)
    rotation = mathutils.Matrix.Rotation(angle, 4, axis)
    inv_origin_map = mathutils.Matrix.Translation(pivot)
    rot_matrix = inv_origin_map @ rotation @ origin_map
    if store:
        transforms.append(rot_matrix)
    return rot_matrix
                             
def transform_selection(transform, store=True):
    """
    Apply transformation matrix on selected vertices.
    Note that blender ops not used. 
    This method can be used for testing the transformation matrix generated with composite transformations.
    Provided transformation matrix can also be stored in transforms if store flag is set.
    """
    global mesh, obj_data, transforms
    if store and len(transform) == 4 and len(transform[0]) == 4:
        transforms.append(transform)

    if len(transform) == 4:
        transform = mathutils.Matrix(transform[0:3])
    selected_verts = list(filter(lambda v: v.select, mesh.verts))
    for v in selected_verts:
        v.co = transform @ mathutils.Vector((*v.co, 1))

    bmesh.update_edit_mesh(obj_data, True)

def compose_transforms(transforms, clear=True):
    """
    Takes a list of 4x4 transformatiom matrices and returns the overall 4x4 matrix. 
    Assumes that the first item of the list is the first transformation and the last item of the list is the last transformation in the sequence.
    Set clear flag to empty processed list.
    """
    composite = mathutils.Matrix.Identity(4)
    for transform in transforms:
        
        if len(transform) != 4 and len(transform[0] != 4):
            return None

        composite = transform @ composite

    if clear:
        transforms.clear()

    return composite

def export_moved_info(path, transforms, objID, clear=True):
    """Takes a single matrix or a list of matrices and stores and returns overall matrix with respect to habitat-sim convention (rotation around x ccw with 90 degrees)"""
    global from_blender_to_habitat, SEG_COLORS

    transform = transforms
    if isinstance(transforms, list):
        transform = compose_transforms(transforms, clear)
    
    transform = from_blender_to_habitat @ transform

    flattened = []
    for row in transform[:3]:
        flattened += list(row)

    info = {
        "color": SEG_COLORS[objID],
        "name": objID,
        "transformation": flattened
    }

    with open(path, "w+") as file:
        json.dump(info, file, indent=4)

    return transform
