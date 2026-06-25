import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import logging

from ..config import DEPTH_MODEL_ID, DEVICE

logger = logging.getLogger("DepthEstimator")

class DepthEstimator:
    def __init__(self):
        logger.info(f"Loading Depth-Anything-V2: {DEPTH_MODEL_ID} on {DEVICE}...")
        self.image_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        self.model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(DEVICE)
        
    def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """
        Runs Depth-Anything-V2 model on the image.
        Returns a 2D numpy array representing the normalized depth values (0.0 to 1.0, closer to 1.0 is closer).
        """
        inputs = self.image_processor(images=image, return_tensors="pt").to(DEVICE)
        
        with torch.inference_mode():
            outputs = self.model(**inputs)
            
        # Post-process to interpolate/upsample depth to original image size
        predicted_depth = outputs.predicted_depth
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        depth_np = prediction.cpu().numpy()
        
        # Normalize to [0, 1] range
        depth_min = depth_np.min()
        depth_max = depth_np.max()
        if depth_max - depth_min > 1e-5:
            normalized_depth = (depth_np - depth_min) / (depth_max - depth_min)
        else:
            normalized_depth = np.zeros_like(depth_np)
            
        logger.info("Successfully calculated and normalized depth map.")
        return normalized_depth
