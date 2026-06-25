import bpy
import json
import os

# Set this to the absolute path of your output_scene folder containing the generated files
SCENE_DIR = r"/path/to/your/output_scene"

def import_scene():
    layout_path = os.path.join(SCENE_DIR, "scene_layout.json")
    if not os.path.exists(layout_path):
        print(f"Error: scene_layout.json not found in {SCENE_DIR}")
        return

    with open(layout_path, "r") as f:
        layout_data = json.load(f)

    # 1. Clear existing objects in the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # 2. Load and set up the background billboard plane
    bg_img_path = os.path.join(SCENE_DIR, "inpainted_background.png")
    if os.path.exists(bg_img_path):
        print("Importing background plane...")
        # Create a plane for the background
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, -1.0))
        bg_plane = bpy.context.active_object
        bg_plane.name = "Background_Backdrop"
        
        # Scale to match original resolution aspect ratio
        width = layout_data["image_resolution"]["width"]
        height = layout_data["image_resolution"]["height"]
        aspect_ratio = width / height
        bg_plane.scale[0] = aspect_ratio * 10.0
        bg_plane.scale[1] = 10.0
        
        # Position background at the back (e.g. Y = 10.0 representing depth)
        bg_plane.location = (0.0, 10.0, 0.0)
        bg_plane.rotation_euler = (1.5708, 0, 0) # Rotate 90 degrees on X-axis

        # Create and assign material with the background texture
        mat = bpy.data.materials.new(name="Background_Material")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clear default nodes
        for node in nodes:
            nodes.remove(node)
            
        # Add Principled BSDF and Output
        output_node = nodes.new(type="ShaderNodeOutputMaterial")
        bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
        tex_node = nodes.new(type="ShaderNodeTexImage")
        
        tex_node.image = bpy.data.images.load(bg_img_path)
        links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
        links.new(tex_node.outputs['Color'], bsdf_node.inputs['Base Color'])
        
        bg_plane.data.materials.append(mat)

    # 3. Import each segmented 3D foreground object
    for obj_data in layout_data.get("objects", []):
        obj_id = obj_data["id"]
        mesh_file = obj_data["mesh_file"]
        mesh_path = os.path.join(SCENE_DIR, mesh_file)
        
        if not os.path.exists(mesh_path):
            print(f"Mesh file {mesh_file} not found, skipping...")
            continue
            
        print(f"Importing object_{obj_id} from {mesh_file}...")
        
        # Import depending on format (.obj or .glb)
        if mesh_file.lower().endswith(".obj"):
            bpy.ops.wm.obj_import(filepath=mesh_path)
        elif mesh_file.lower().endswith(".glb"):
            bpy.ops.import_scene.gltf(filepath=mesh_path)
            
        imported_obj = bpy.context.selected_objects[0]
        imported_obj.name = f"Foreground_Object_{obj_id}"
        
        # Apply transformation calculated from depth map
        pos = obj_data["transform"]["position"]
        scale = obj_data["transform"]["scale"]
        
        imported_obj.location = (pos["x"], pos["y"], pos["z"])
        imported_obj.scale = (scale["x"], scale["y"], scale["z"])

    print("Scene compilation completed in Blender successfully!")

# Run the import function
import_scene()
