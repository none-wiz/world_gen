import os
import argparse
import logging
from PIL import Image
import numpy as np

# Configure Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("PipelineOrchestrator")

# Import Modules
from config import OUTPUT_DIR, DEVICE, INPAINT_MODEL_ID, INPAINT_DTYPE, MESH_GENERATOR_TYPE
from utils.memory import clear_gpu_memory, MemoryGuard
from utils.image_processing import extract_masked_object, crop_and_pad_object, run_inpainting_pipeline
from modules.segmenter import ObjectSegmenter
from modules.depth_estimator import DepthEstimator
from modules.mesh_generator import MeshGenerator
from modules.scene_compiler import SceneCompiler

def parse_args():
    parser = argparse.ArgumentParser(description="3D World Preprocessing Pipeline")
    parser.add_argument(
        "--image", 
        type=str, 
        required=True, 
        help="Path to the input 2D image."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=OUTPUT_DIR, 
        help="Directory to save generated 3D assets and layout."
    )
    return parser.parse_args()

def run_pipeline(image_path: str, output_dir: str):
    logger.info(f"Starting 3D World Preprocessing Pipeline for image: {image_path}")
    
    # 1. Load and verify input image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    
    input_image = Image.open(image_path).convert("RGB")
    w, h = input_image.size
    logger.info(f"Loaded image. Resolution: {w}x{h}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # STAGE 1: Semantic Segmentation (SAM 2)
    # -------------------------------------------------------------------------
    logger.info("--- STAGE 1: Semantic Segmentation ---")
    with MemoryGuard():
        segmenter = ObjectSegmenter()
        segments = segmenter.generate_masks(input_image)
        
        # Save individual transparent objects
        for seg in segments:
            obj_id = seg["id"]
            mask = seg["mask"]
            
            # Extract transparent foreground object
            transparent_obj = extract_masked_object(input_image, mask)
            # Pad to square
            padded_obj = crop_and_pad_object(transparent_obj)
            
            obj_path = os.path.join(output_dir, f"object_{obj_id}.png")
            padded_obj.save(obj_path, format="PNG")
            logger.info(f"Saved segmented foreground object to: {obj_path}")
            
        # Clean up segmenter explicitly
        del segmenter
        
    # -------------------------------------------------------------------------
    # STAGE 2: Depth Map Estimation
    # -------------------------------------------------------------------------
    logger.info("--- STAGE 2: Depth Estimation ---")
    with MemoryGuard():
        depth_estimator = DepthEstimator()
        depth_map = depth_estimator.estimate_depth(input_image)
        
        # Save depth map as visualization image
        depth_viz = (depth_map * 255).astype(np.uint8)
        depth_img = Image.fromarray(depth_viz)
        depth_img.save(os.path.join(output_dir, "scene_depth.png"))
        logger.info("Saved scene depth map visualization.")
        
        del depth_estimator

    # -------------------------------------------------------------------------
    # STAGE 3: Inpainting / Background Fill
    # -------------------------------------------------------------------------
    logger.info("--- STAGE 3: Background Inpainting ---")
    if len(segments) > 0:
        with MemoryGuard():
            from diffusers import StableDiffusionInpaintPipeline
            
            # Combine all foreground masks to get overall background hole mask
            combined_mask = np.zeros((h, w), dtype=bool)
            for seg in segments:
                combined_mask = np.logical_or(combined_mask, seg["mask"])
                
            logger.info(f"Loading Inpainting pipeline: {INPAINT_MODEL_ID}...")
            try:
                inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    INPAINT_MODEL_ID,
                    torch_dtype=INPAINT_DTYPE
                ).to(DEVICE)
                
                # Perform inpainting
                inpainted_bg = run_inpainting_pipeline(inpaint_pipe, input_image, combined_mask)
                del inpaint_pipe
            except Exception as e:
                logger.warning(
                    f"Failed to run Stable Diffusion inpainting: {e}. "
                    "Falling back to basic PIL blur inpainting to keep the pipeline alive."
                )
                from PIL import ImageFilter
                mask_uint8 = (combined_mask * 255).astype(np.uint8)
                mask_image = Image.fromarray(mask_uint8, mode="L")
                
                # Blend blurred background to fill the foreground holes roughly
                blurred = input_image.filter(ImageFilter.GaussianBlur(radius=20))
                inpainted_bg = Image.composite(blurred, input_image, mask_image)
            
            bg_path = os.path.join(output_dir, "inpainted_background.png")
            inpainted_bg.save(bg_path, format="PNG")
            logger.info(f"Saved background image to: {bg_path}")
    else:
        logger.warning("No foreground segments found. Skipping background inpainting.")

    # -------------------------------------------------------------------------
    # STAGE 4: Single-Image-to-3D Generation
    # -------------------------------------------------------------------------
    logger.info("--- STAGE 4: 3D Mesh Generation ---")
    with MemoryGuard():
        mesh_gen = MeshGenerator(generator_type=MESH_GENERATOR_TYPE)
        
        for seg in segments:
            obj_id = seg["id"]
            img_path = os.path.join(output_dir, f"object_{obj_id}.png")
            obj_img = Image.open(img_path)
            
            output_mesh_path = os.path.join(output_dir, f"object_{obj_id}.glb")
            mesh_gen.generate_mesh(obj_img, output_mesh_path)
            
        del mesh_gen

    # -------------------------------------------------------------------------
    # STAGE 5: Scene Metadata Compilation
    # -------------------------------------------------------------------------
    logger.info("--- STAGE 5: Scene Compilation ---")
    compiler = SceneCompiler(output_dir)
    layout_file = compiler.compile_and_export(input_image.size, segments, depth_map)
    
    logger.info("--- Pipeline Completed Successfully! ---")
    logger.info(f"All assets and scene configurations saved to: {output_dir}")
    logger.info(f"Blender import metadata: {layout_file}")

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.image, args.output_dir)
