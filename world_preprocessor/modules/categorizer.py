import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import logging
import json
import os
from typing import Dict

from config import VLM_MODEL_ID, VLM_DTYPE, DEVICE

logger = logging.getLogger("ObjectCategorizer")

class ObjectCategorizer:
    def __init__(self):
        logger.info(f"Loading Visual Language Model: {VLM_MODEL_ID} on {DEVICE}...")
        self.model = None
        self.processor = None
        
        try:
            self.processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                VLM_MODEL_ID,
                torch_dtype=VLM_DTYPE,
                device_map="auto" if DEVICE == "cuda" else None
            )
            if DEVICE == "cuda" and not hasattr(self.model, "device"):
                self.model = self.model.to(DEVICE)
        except Exception as e:
            logger.warning(
                f"Failed to load VLM model {VLM_MODEL_ID}: {e}. "
                "The categorizer will fall back to using generic placeholder labels."
            )

    def categorize_object(self, image: Image.Image, object_id: int) -> str:
        """
        Takes an image of a segmented foreground object and returns a short, descriptive name/tag.
        """
        if self.model is None or self.processor is None:
            return f"segmented_object_{object_id}"
            
        try:
            # Prepare instructions
            prompt_text = "Describe this segmented object briefly in 2-5 words. For example: 'a green pine tree', 'a brown wooden chair', 'a small red car'."
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Process inputs
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            ).to(DEVICE)
            
            if VLM_DTYPE == torch.float16:
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
            
            # Generate description
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=20)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            # Sanitize description
            clean_desc = output_text.replace("\n", " ").replace('"', '').strip()
            logger.info(f"Categorized object {object_id}: '{clean_desc}'")
            return clean_desc
            
        except Exception as e:
            logger.error(f"Error during VLM categorization of object {object_id}: {e}. Returning fallback.")
            return f"object_{object_id}"

    def run_categorization(self, output_dir: str, segments_list: list) -> Dict[str, str]:
        """
        Loops through all segment images and saves their descriptions in object_descriptions.json.
        """
        descriptions = {}
        
        for seg in segments_list:
            obj_id = seg["id"]
            img_path = os.path.join(output_dir, f"object_{obj_id}.png")
            
            if not os.path.exists(img_path):
                logger.warning(f"Segment image not found for id {obj_id}, skipping categorization.")
                continue
                
            img = Image.open(img_path).convert("RGB")
            desc = self.categorize_object(img, obj_id)
            descriptions[f"object_{obj_id}"] = desc
            
        # Write to JSON file
        desc_file = os.path.join(output_dir, "object_descriptions.json")
        with open(desc_file, "w") as f:
            json.dump(descriptions, f, indent=4)
            
        logger.info(f"Successfully saved object descriptions to {desc_file}")
        return descriptions
