import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor
import logging
from typing import List, Dict, Any

from config import SAM2_MODEL_ID, SAM2_DTYPE, DEVICE

logger = logging.getLogger("Segmenter")

class ObjectSegmenter:
    def __init__(self):
        logger.info(f"Loading SAM 2 Model: {SAM2_MODEL_ID} on {DEVICE}...")
        self.processor = Sam2Processor.from_pretrained(SAM2_MODEL_ID)
        self.model = Sam2Model.from_pretrained(
            SAM2_MODEL_ID, 
            torch_dtype=SAM2_DTYPE
        ).to(DEVICE)
        
    def generate_masks(self, image: Image.Image, grid_size: int = 16) -> List[Dict[str, Any]]:
        """
        Generates individual object masks by prompting SAM 2 with a grid of point anchors.
        Returns a list of dictionaries containing the binary masks and bounding boxes.
        """
        w, h = image.size
        
        # Build grid points
        xs = np.linspace(w * 0.1, w * 0.9, grid_size)
        ys = np.linspace(h * 0.1, h * 0.9, grid_size)
        
        input_points = []
        input_labels = []
        for x in xs:
            for y in ys:
                input_points.append([[[x, y]]])
                input_labels.append([[1]])
                
        # To avoid overloading memory, run inference in batches of point prompts
        batch_size = 16
        all_masks = []
        
        for i in range(0, len(input_points), batch_size):
            batch_pts = input_points[i:i + batch_size]
            batch_lbls = input_labels[i:i + batch_size]
            
            # Prepare inputs
            # The Sam2Processor expects list of points for the image
            inputs = self.processor(
                images=[image] * len(batch_pts),
                input_points=batch_pts,
                input_labels=batch_lbls,
                return_tensors="pt"
            ).to(DEVICE)
            
            # Use appropriate dtype matching config
            if SAM2_DTYPE == torch.float16:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
                
            with torch.inference_mode():
                outputs = self.model(**inputs)
                
            original_sizes = inputs.get("original_sizes", None)
            reshaped_input_sizes = inputs.get("reshaped_input_sizes", None)
            
            try:
                # Use kwargs to avoid positional mismatch (like overwriting mask_threshold with None)
                post_kwargs = {}
                if original_sizes is not None:
                    post_kwargs["original_sizes"] = original_sizes
                if reshaped_input_sizes is not None:
                    post_kwargs["reshaped_input_sizes"] = reshaped_input_sizes
                
                masks = self.processor.post_process_masks(
                    outputs.pred_masks,
                    **post_kwargs
                )
            except Exception as e:
                logger.warning(f"SAM2 post_process_masks failed: {e}. Running manual interpolation fallback.")
                # Manual resize fallback using torch interpolation
                masks = []
                for idx, pred_mask in enumerate(outputs.pred_masks):
                    # pred_mask shape: [num_masks, H, W]
                    target_h, target_w = image.size[1], image.size[0]
                    if original_sizes is not None:
                        try:
                            target_h = int(original_sizes[idx][0])
                            target_w = int(original_sizes[idx][1])
                        except Exception:
                            pass
                    
                    # Interpolate [num_masks, H, W] -> [1, num_masks, target_h, target_w] -> [num_masks, target_h, target_w]
                    upsampled = torch.nn.functional.interpolate(
                        pred_mask.unsqueeze(0).float(),
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0)
                    masks.append(upsampled > 0.0)
            
            for mask_list in masks:
                # Get the highest confidence mask for each point prompt
                # mask_list is of shape [num_masks, H, W]
                # Typically, SAM returns 3 scales of masks; we choose the first or argmax score
                for sub_m in mask_list:
                    # Let's convert to binary numpy array
                    binary_mask = sub_m[0].cpu().numpy() > 0.0
                    all_masks.append(binary_mask)
                    
        # Filter and deduplicate masks
        unique_masks = []
        for mask in all_masks:
            # Check if mask has too few or too many pixels (ignore background or noise)
            pixel_count = np.sum(mask)
            total_pixels = w * h
            if pixel_count < 0.005 * total_pixels or pixel_count > 0.85 * total_pixels:
                continue
                
            # Check overlap with existing unique masks (IoU deduplication)
            is_duplicate = False
            for u_mask in unique_masks:
                intersection = np.logical_and(mask, u_mask).sum()
                union = np.logical_or(mask, u_mask).sum()
                iou = intersection / union if union > 0 else 0
                if iou > 0.65:  # High overlap indicates duplicate detection
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_masks.append(mask)
                
        results = []
        for idx, mask in enumerate(unique_masks):
            # Calculate simple bounding box [ymin, xmin, ymax, xmax]
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            results.append({
                "id": idx,
                "mask": mask,
                "bbox": [int(ymin), int(xmin), int(ymax), int(xmax)]
            })
            
        logger.info(f"Segmented {len(results)} individual objects from the image.")
        return results

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor
import logging
from typing import List, Dict, Any
from tqdm import tqdm

from config import SAM2_DTYPE, DEVICE

# Your updated Model ID configuration
SAM3_MODEL_ID = "facebook/sam3"

logger = logging.getLogger("Segmenter")

class ObjectSegmenterV3:
    def __init__(self):
        logger.info(f"Initializing SAM 3 ObjectSegmenter")
        logger.info(f"Target Compute Device: {DEVICE}")
        if torch.cuda.is_available() and "cuda" in str(DEVICE):
            logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(DEVICE)}")
            logger.info(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(DEVICE) / 1024**2:.2f} MB")
            
        logger.info(f"Loading SAM 3 Model: {SAM3_MODEL_ID} using dtype={SAM2_DTYPE}...")
        
        # --- FIX: Pass token=True to authorize the gated repo download ---
        self.processor = Sam3Processor.from_pretrained(
            SAM3_MODEL_ID, 
            token=True
        )
        self.model = Sam3Model.from_pretrained(
            SAM3_MODEL_ID, 
            torch_dtype=SAM2_DTYPE,
            token=True
        ).to(DEVICE)
        # -----------------------------------------------------------------
        
        self.model.eval()
        
    def generate_masks(self, image: Image.Image, text_queries: List[str] = None, score_threshold: float = 0.2, mask_threshold: float = 0.2) -> List[Dict[str, Any]]:
        """
        Generates individual object masks using open-vocabulary SAM 3 text queries.
        
        Args:
            image: PIL Input image.
            text_queries: List of concept strings to search for (e.g., ["chair", "mug", "person"]).
                         Defaults to ["object"] for general extraction.
        """
        w, h = image.size
        all_masks = []
        all_scores = []
        
        logger.info(f"Running SAM 3 Promptable Concept Segmentation for queries: {text_queries}")
        if text_queries is None: 
          text_queries = ["object"]
        # Process each semantic concept query
        for query in tqdm(text_queries, desc="Processing SAM3 Text Queries", unit="concept"):
            inputs = self.processor(
                images=image, 
                text=query, 
                return_tensors="pt"
            ).to(DEVICE)
            
            # Match precision type
            if SAM2_DTYPE in [torch.float16, torch.bfloat16]:
                inputs["pixel_values"] = inputs["pixel_values"].to(SAM2_DTYPE)
                
            with torch.inference_mode():
                outputs = self.model(**inputs)
                
            # SAM 3 native post-processing scales and threshold filters the results automatically
            processed_results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=score_threshold,
                mask_threshold=mask_threshold,
                target_sizes=[[h, w]]
            )[0]
            
            # Extract returned instances
            pred_masks = processed_results["masks"]     # Shape: [num_instances, H, W]
            pred_scores = processed_results["scores"]   # Shape: [num_instances]
            
            for idx in range(pred_masks.shape[0]):
                binary_mask = pred_masks[idx].cpu().numpy()
                all_masks.append(binary_mask)
                all_scores.append(float(pred_scores[idx]))

        # Deduplicate overlapping boundaries across different target queries
        unique_masks = []
        unique_scores = []
        
        for mask, score in tqdm(zip(all_masks, all_scores), total=len(all_masks), desc="De-duplicating cross-query instances", unit="mask"):
            pixel_count = np.sum(mask)
            total_pixels = w * h
            if pixel_count < 0.001 * total_pixels or pixel_count > 0.85 * total_pixels:
                continue
                
            is_duplicate = False
            for u_mask in unique_masks:
                intersection = np.logical_and(mask, u_mask).sum()
                union = np.logical_or(mask, u_mask).sum()
                iou = intersection / union if union > 0 else 0
                if iou > 0.65:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_masks.append(mask)
                unique_scores.append(score)
                
        results = []
        for idx, (mask, score) in enumerate(zip(unique_masks, unique_scores)):
            # Calculate final bounding boxes from filtered arrays
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            results.append({
                "id": idx,
                "mask": mask,
                "confidence": score,
                "bbox": [int(ymin), int(xmin), int(ymax), int(xmax)]
            })
            
        logger.info(f"SAM 3 extracted {len(results)} distinct objects from concept inquiries.")
        return results