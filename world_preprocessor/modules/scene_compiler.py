import json
import os
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger("SceneCompiler")

class SceneCompiler:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.layout_file = os.path.join(output_dir, "scene_layout.json")

    def compile_and_export(
        self, 
        image_size: tuple, 
        segments: List[Dict[str, Any]], 
        depth_map: np.ndarray
    ):
        """
        Computes 3D transform metrics (X, Y, Z coords, scaling factor) for each segment.
        Writes a unified scene_layout.json file for Blender importing.
        """
        img_w, img_h = image_size
        scene_objects = []

        for seg in segments:
            obj_id = seg["id"]
            mask = seg["mask"]
            bbox = seg["bbox"]  # [ymin, xmin, ymax, xmax]
            
            # Calculate 2D centroid (pixels)
            ymin, xmin, ymax, xmax = bbox
            center_x = (xmin + xmax) / 2.0
            center_y = (ymin + ymax) / 2.0
            
            # Compute normalized coordinates (centered at 0.0, range -0.5 to 0.5)
            norm_x = (center_x / img_w) - 0.5
            # Flip Y for standard Cartesian 3D coordinates (Blender uses +Z up, +Y forward/depth or similar)
            norm_y = 0.5 - (center_y / img_h)
            
            # Calculate depth value for the object (median of the object's masked area)
            obj_depths = depth_map[mask]
            median_depth = float(np.median(obj_depths)) if len(obj_depths) > 0 else 0.5
            
            # Normalized scale relative to the scene (width and height ratios)
            scale_x = (xmax - xmin) / img_w
            scale_y = (ymax - ymin) / img_h
            
            # Determine filenames (check if high-quality .glb exists, else use .obj fallback)
            glb_filename = f"object_{obj_id}.glb"
            if os.path.exists(os.path.join(self.output_dir, glb_filename)):
                mesh_filename = glb_filename
            else:
                mesh_filename = f"object_{obj_id}.obj"
            texture_filename = f"object_{obj_id}_tex.png"
            
            # Prepare metadata entry
            obj_data = {
                "id": obj_id,
                "mesh_file": mesh_filename,
                "texture_file": texture_filename,
                "transform": {
                    "position": {
                        "x": round(norm_x * 10.0, 3),      # Scale up coordinates for Blender visibility
                        "y": round(median_depth * 10.0, 3), # Depth maps to Y or Z axis depending on setup
                        "z": round(norm_y * 10.0, 3)       # Height maps to Z
                    },
                    "scale": {
                        "x": round(scale_x * 5.0, 3),
                        "y": round(scale_y * 5.0, 3),
                        "z": round((scale_x + scale_y) * 2.5, 3)  # Approximate depth thickness
                    },
                    "rotation": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0
                    }
                },
                "bounds_2d": {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                },
                "mean_depth": round(median_depth, 4)
            }
            scene_objects.append(obj_data)
            
        # Export layout JSON
        scene_layout = {
            "image_resolution": {
                "width": img_w,
                "height": img_h
            },
            "objects": scene_objects
        }
        
        with open(self.layout_file, "w") as f:
            json.dump(scene_layout, f, indent=4)
            
        logger.info(f"Successfully compiled and saved scene layout to: {self.layout_file}")
        return self.layout_file
