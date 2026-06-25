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
        
    def generate_masks(self, image: Image.Image, grid_size: int = 8) -> List[Dict[str, Any]]:
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
            
            masks = self.processor.post_process_masks(
                outputs.pred_masks,
                original_sizes,
                reshaped_input_sizes
            )
            
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
