import numpy as np
from dataclasses import dataclass
import bpy
import blender_plots as bplt
from blender_plots import blender_utils as bu


def setup_scene(floor_z=None):
    # delete default cube
    if "Cube" in bpy.data.objects:
        bu.delete(bpy.data.objects["Cube"])

    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    if "Sun" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Sun"])
    bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.data.objects["Sun"].data.energy = 1.
    bpy.data.objects["Sun"].data.angle = 0
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    bpy.context.scene.render.engine = 'CYCLES'

    if floor_z is not None:
        floor_size = 500
        floor = bplt.Scatter([0, 0, floor_z], marker_type='cubes', size=(floor_size, floor_size, 0.1), name='floor')
        floor.base_object.is_shadow_catcher = True

def srgb_to_linearrgb(c):
    if   c < 0:       return 0
    elif c < 0.04045: return c/12.92
    else:             return ((c+0.055)/1.055)**2.4

def hex_to_rgb(h,alpha=1):
    # from: https://blender.stackexchange.com/questions/153094/blender-2-8-python-how-to-set-material-color-using-hex-value-instead-of-rgb
    r = (h & 0xff0000) >> 16
    g = (h & 0x00ff00) >> 8
    b = (h & 0x0000ff)
    return tuple([srgb_to_linearrgb(c/0xff) for c in (r,g,b)] + [alpha])


def set_color(mesh, color):
    if len(color) == 3:
        color = [*color, 1.]
    mesh.materials.append(bpy.data.materials.new("color"))
    mesh.materials[0].use_nodes = True
    mesh.materials[0].node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = color

def plot_cameras(poses, intrinsics, heights, widths, frustum_color=None, fill_color=None, name="", image_depth=0.25):
    if frustum_color is None:
        frustum_color = Colors.orange1
    if fill_color is None:
        fill_color = Colors.orange2

    frustum_points = np.vstack([
        np.einsum('ij,...j->...i', poses[i].R @ np.linalg.inv(intrinsics[i]), np.array([
            [0, heights[i], 1],
            [widths[i], heights[i], 1],
            [0, 0, 1],
            [widths[i], 0, 1],
            [0, 0, 0]
        ]) * image_depth) + poses[i].t
        for i in range(len(intrinsics))
    ])


    frustum_edges = np.vstack([
        np.array([
            [0, 1],
            [1, 3],
            [3, 2],
            [2, 0],
            [0, 4],
            [1, 4],
            [2, 4],
            [3, 4]
        ]) + i * 5
        for i in range(len(intrinsics))
    ])

    frustum_faces = np.vstack([
        np.array([
            [0, 1, 4],
            [1, 3, 4],
            [3, 2, 4],
            [2, 0, 4],
        ]) + i * 5
        for i in range(len(intrinsics))
    ])

    mesh = bpy.data.meshes.new(f"frustum_{name}")
    mesh.from_pydata(frustum_points, frustum_edges, frustum_faces) #, frustum_edges, frustum_faces)
    frustum = bu.new_empty(f"frustum_{name}", mesh)
    modifier = bu.add_modifier(frustum, "WIREFRAME", use_crease=True, crease_weight=0.6, thickness=0.03, use_boundary=True)

    fill_faces = list(frustum_faces)
    for i in range(len(intrinsics)):
        fill_faces.append([
            i * 5 + 0,
            i * 5 + 1,
            i * 5 + 3,
            i * 5 + 2,
        ])

    mesh_fill = bpy.data.meshes.new(f"fill_{name}")
    mesh_fill.from_pydata(np.vstack([frustum_points, np.zeros(3)]), frustum_edges, fill_faces)
    fill = bu.new_empty(f"fill_{name}", mesh_fill)

    set_color(frustum.data, frustum_color)
    set_color(fill.data, fill_color)

def make_line(start, end, width=0.05, name="line", color=None):
    line_mesh = bpy.data.meshes.new(name)
    line_mesh.from_pydata([start, end], [[0, 1]], [])
    line = bplt.blender_utils.new_empty(name, object_data=line_mesh)
    
    bplt.blender_utils.add_modifier(line, "SKIN")
    bplt.blender_utils.add_modifier(line, "SUBSURF", levels=3, render_levels=3)
    
    line.data.skin_vertices[''].data[0].radius = (width, width)
    line.data.skin_vertices[''].data[1].radius = (width, width)
    
    if color is not None:
        set_color(line_mesh, color)
    return line

@dataclass
class Colors:
    orange1 = hex_to_rgb(0xff9a00)
    orange2 = hex_to_rgb(0xff5d00)
    blue1 = hex_to_rgb(0x00a2ff)
    blue2 = hex_to_rgb(0x0065ff)
    white = [1, 1, 1]
