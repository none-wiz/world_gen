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

    # 4. Apply Animation Schedule (if animation_schedule.json exists)
    schedule_path = os.path.join(SCENE_DIR, "animation_schedule.json")
    if os.path.exists(schedule_path):
        print("Applying animation schedule...")
        with open(schedule_path, "r") as f:
            schedule = json.load(f)
            
        total_frames = schedule.get("total_frames", 250)
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = total_frames
        
        # Ensure default camera and light exist for keyframing
        if not any(obj.type == 'LIGHT' for obj in bpy.data.objects):
            bpy.ops.object.light_add(type='SUN', radius=1.0, location=(0.0, 0.0, 10.0))
            sun = bpy.context.active_object
            sun.name = "Light"
            
        if not any(obj.type == 'CAMERA' for obj in bpy.data.objects):
            bpy.ops.object.camera_add(location=(0.0, -15.0, 2.0))
            cam = bpy.context.active_object
            cam.name = "Camera"
            cam.rotation_euler = (1.4835, 0.0, 0.0)

        for event in schedule.get("events", []):
            frame = event.get("frame", 1)
            target_name = event.get("target")
            action_type = event.get("action")
            params = event.get("params", {})
            
            obj = bpy.data.objects.get(target_name)
            if not obj:
                print(f"Animation Warning: Target '{target_name}' not found. Skipping.")
                continue
                
            bpy.context.scene.frame_set(frame)
            
            # Camera Movements
            if action_type == "MOVE_UP":
                obj.location[2] += params.get("value", 1.0)
                obj.keyframe_insert(data_path="location", frame=frame)
            elif action_type == "MOVE_DOWN":
                obj.location[2] -= params.get("value", 1.0)
                obj.keyframe_insert(data_path="location", frame=frame)
            elif action_type == "ZOOM_IN" and obj.type == 'CAMERA':
                obj.data.lens += params.get("value", 5.0)
                obj.data.keyframe_insert(data_path="lens", frame=frame)
            elif action_type == "ZOOM_OUT" and obj.type == 'CAMERA':
                obj.data.lens -= params.get("value", 5.0)
                obj.data.keyframe_insert(data_path="lens", frame=frame)
            elif action_type == "PAN_LEFT":
                obj.location[0] -= params.get("value", 1.0)
                obj.keyframe_insert(data_path="location", frame=frame)
            elif action_type == "PAN_RIGHT":
                obj.location[0] += params.get("value", 1.0)
                obj.keyframe_insert(data_path="location", frame=frame)
                
            # Object/Character Movements
            elif action_type == "MOVE":
                dest = params.get("destination", [0.0, 0.0, 0.0])
                obj.location = (dest[0], dest[1], dest[2])
                obj.keyframe_insert(data_path="location", frame=frame)
            elif action_type == "ROTATE":
                rot = params.get("rotation", [0.0, 0.0, 0.0])
                obj.rotation_euler = (rot[0], rot[1], rot[2])
                obj.keyframe_insert(data_path="rotation_euler", frame=frame)
            elif action_type == "JUMP":
                height = params.get("height", 2.0)
                orig_z = obj.location[2]
                obj.keyframe_insert(data_path="location", frame=frame - 10)
                obj.location[2] = orig_z + height
                obj.keyframe_insert(data_path="location", frame=frame)
                obj.location[2] = orig_z
                obj.keyframe_insert(data_path="location", frame=frame + 10)
                
            # Character Animations (blinking eye, moving ear)
            elif action_type == "BLINK_EYE":
                if obj.data.shape_keys:
                    key_blocks = obj.data.shape_keys.key_blocks
                    if "Eye_Blink" in key_blocks:
                        key_blocks["Eye_Blink"].value = 0.0
                        key_blocks["Eye_Blink"].keyframe_insert(data_path="value", frame=frame - 2)
                        key_blocks["Eye_Blink"].value = 1.0
                        key_blocks["Eye_Blink"].keyframe_insert(data_path="value", frame=frame)
                        key_blocks["Eye_Blink"].value = 0.0
                        key_blocks["Eye_Blink"].keyframe_insert(data_path="value", frame=frame + 2)
            elif action_type == "MOVE_EAR":
                if obj.data.shape_keys:
                    key_blocks = obj.data.shape_keys.key_blocks
                    if "Ear_Move" in key_blocks:
                        key_blocks["Ear_Move"].value = 1.0
                        key_blocks["Ear_Move"].keyframe_insert(data_path="value", frame=frame)
                        key_blocks["Ear_Move"].value = 0.0
                        key_blocks["Ear_Move"].keyframe_insert(data_path="value", frame=frame + 10)
                        
            # Light Intensity adjustments
            elif action_type == "BRIGHTEN" and obj.type == 'LIGHT':
                obj.data.energy *= params.get("factor", 1.5)
                obj.data.keyframe_insert(data_path="energy", frame=frame)
            elif action_type == "DARKEN" and obj.type == 'LIGHT':
                obj.data.energy *= params.get("factor", 0.5)
                obj.data.keyframe_insert(data_path="energy", frame=frame)
            elif action_type == "TURN_OFF" and obj.type == 'LIGHT':
                obj.data.energy = 0.0
                obj.data.keyframe_insert(data_path="energy", frame=frame)
            elif action_type == "TURN_ON" and obj.type == 'LIGHT':
                obj.data.energy = params.get("intensity", 100.0)
                obj.data.keyframe_insert(data_path="energy", frame=frame)

    print("Scene compilation completed in Blender successfully!")

# Run the import function
import_scene()
