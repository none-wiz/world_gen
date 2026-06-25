import json
import os
import logging
from enum import Enum

logger = logging.getLogger("SceneRenderer")

# 1. Animation Action Enums
class CameraMovement(Enum):
    MOVE_UP = "MOVE_UP"
    MOVE_DOWN = "MOVE_DOWN"
    ZOOM_IN = "ZOOM_IN"
    ZOOM_OUT = "ZOOM_OUT"
    PAN_LEFT = "PAN_LEFT"
    PAN_RIGHT = "PAN_RIGHT"

class ObjectMovement(Enum):
    MOVE = "MOVE"
    ROTATE = "ROTATE"
    JUMP = "JUMP"
    GLIDE = "GLIDE"
    SCALE_UP = "SCALE_UP"
    SCALE_DOWN = "SCALE_DOWN"

class LightingAction(Enum):
    BRIGHTEN = "BRIGHTEN"
    DARKEN = "DARKEN"
    TURN_ON = "TURN_ON"
    TURN_OFF = "TURN_OFF"

class CharacterAnimation(Enum):
    BLINK_EYE = "BLINK_EYE"
    MOVE_EAR = "MOVE_EAR"
    JUMP = "JUMP"
    ROTATE = "ROTATE"
    MOVE = "MOVE"

# 2. Blender animation scheduler execution
def apply_animation_schedule(schedule_path: str):
    """
    Parses the animation_schedule.json and runs inside Blender to keyframe animations on objects/cameras/lights.
    """
    try:
        import bpy
    except ImportError:
        logger.warning(
            "bpy library not found! The renderer module must be run inside Blender's python environment "
            "to keyframe animations. Skipping Blender execution."
        )
        return

    if not os.path.exists(schedule_path):
        logger.error(f"Animation schedule JSON not found: {schedule_path}")
        return

    with open(schedule_path, "r") as f:
        schedule = json.load(f)

    # Set frame range based on schedule duration
    total_frames = schedule.get("total_frames", 250)
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = total_frames

    # Parse and execute animations frame by frame
    for event in schedule.get("events", []):
        frame = event.get("frame", 1)
        target_name = event.get("target")  # e.g., "Camera", "Foreground_Object_0", "Light"
        action_type = event.get("action")  # e.g., "MOVE_UP", "JUMP", "BRIGHTEN", "BLINK_EYE"
        params = event.get("params", {})   # coordinates, intensities, etc.

        # Find the object in Blender
        obj = bpy.data.objects.get(target_name)
        if not obj:
            logger.warning(f"Blender object not found: {target_name}. Skipping event.")
            continue

        bpy.context.scene.frame_set(frame)

        # -------------------------------------------------------------
        # CAMERA MOVEMENTS
        # -------------------------------------------------------------
        if action_type in [e.value for e in CameraMovement]:
            if action_type == CameraMovement.MOVE_UP.value:
                val = params.get("value", 1.0)
                obj.location[2] += val
            elif action_type == CameraMovement.MOVE_DOWN.value:
                val = params.get("value", 1.0)
                obj.location[2] -= val
            elif action_type == CameraMovement.ZOOM_IN.value:
                # Adjust camera focal length if target is a Camera object
                if obj.type == 'CAMERA':
                    obj.data.lens += params.get("value", 5.0)
                    obj.data.keyframe_insert(data_path="lens", frame=frame)
            elif action_type == CameraMovement.ZOOM_OUT.value:
                if obj.type == 'CAMERA':
                    obj.data.lens -= params.get("value", 5.0)
                    obj.data.keyframe_insert(data_path="lens", frame=frame)
            elif action_type == CameraMovement.PAN_LEFT.value:
                val = params.get("value", 1.0)
                obj.location[0] -= val
            elif action_type == CameraMovement.PAN_RIGHT.value:
                val = params.get("value", 1.0)
                obj.location[0] += val

            obj.keyframe_insert(data_path="location", frame=frame)

        # -------------------------------------------------------------
        # OBJECT / CHARACTER MOVEMENTS
        # -------------------------------------------------------------
        elif action_type in [e.value for e in ObjectMovement] or action_type in [e.value for e in CharacterAnimation]:
            if action_type == ObjectMovement.MOVE.value or action_type == CharacterAnimation.MOVE.value:
                dest = params.get("destination", [0.0, 0.0, 0.0])
                obj.location = (dest[0], dest[1], dest[2])
                obj.keyframe_insert(data_path="location", frame=frame)
                
            elif action_type == ObjectMovement.ROTATE.value or action_type == CharacterAnimation.ROTATE.value:
                rot = params.get("rotation", [0.0, 0.0, 0.0])  # Euler angles
                obj.rotation_euler = (rot[0], rot[1], rot[2])
                obj.keyframe_insert(data_path="rotation_euler", frame=frame)
                
            elif action_type == ObjectMovement.JUMP.value or action_type == CharacterAnimation.JUMP.value:
                # Keyframe a jump sequence: frame-10 -> rise, frame -> peak, frame+10 -> land
                height = params.get("height", 2.0)
                orig_z = obj.location[2]
                
                # Setup jump arc keyframes
                obj.keyframe_insert(data_path="location", frame=frame - 10)
                obj.location[2] = orig_z + height
                obj.keyframe_insert(data_path="location", frame=frame)
                obj.location[2] = orig_z
                obj.keyframe_insert(data_path="location", frame=frame + 10)
                
            elif action_type == CharacterAnimation.BLINK_EYE.value:
                # Simulates shape key animation for blinking (if shape keys exist on mesh)
                if obj.data.shape_keys:
                    key_blocks = obj.data.shape_keys.key_blocks
                    if "Eye_Blink" in key_blocks:
                        key_blocks["Eye_Blink"].value = 0.0
                        key_blocks["Eye_Blink"].keyframe_insert(data_path="value", frame=frame - 2)
                        key_blocks["Eye_Blink"].value = 1.0
                        key_blocks["Eye_Blink"].keyframe_insert(data_path="value", frame=frame)
                        key_blocks["Eye_Blink"].value = 0.0
                        key_blocks["Eye_Blink"].keyframe_insert(data_path="value", frame=frame + 2)
                        
            elif action_type == CharacterAnimation.MOVE_EAR.value:
                # Simulates rotation or shape key changes for moving character ears
                if obj.data.shape_keys:
                    key_blocks = obj.data.shape_keys.key_blocks
                    if "Ear_Move" in key_blocks:
                        key_blocks["Ear_Move"].value = 1.0
                        key_blocks["Ear_Move"].keyframe_insert(data_path="value", frame=frame)
                        key_blocks["Ear_Move"].value = 0.0
                        key_blocks["Ear_Move"].keyframe_insert(data_path="value", frame=frame + 10)

        # -------------------------------------------------------------
        # LIGHTING ACTIONS
        # -------------------------------------------------------------
        elif action_type in [e.value for e in LightingAction]:
            # Apply only if the target is a light source
            if obj.type == 'LIGHT':
                light_data = obj.data
                if action_type == LightingAction.BRIGHTEN.value:
                    light_data.energy *= params.get("factor", 1.5)
                elif action_type == LightingAction.DARKEN.value:
                    light_data.energy *= params.get("factor", 0.5)
                elif action_type == LightingAction.TURN_OFF.value:
                    light_data.energy = 0.0
                elif action_type == LightingAction.TURN_ON.value:
                    light_data.energy = params.get("intensity", 100.0)

                light_data.keyframe_insert(data_path="energy", frame=frame)

    logger.info("Successfully scheduled animations in Blender.")
