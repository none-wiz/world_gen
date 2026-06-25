import numpy as np
from PIL import Image, ImageOps
import torch

def extract_masked_object(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    Extracts an object from an image using a binary mask, returning a transparent PNG.
    """
    # Ensure image is in RGBA mode
    rgba_image = image.convert("RGBA")
    
    # Convert mask to 0-255 uint8 format
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_uint8, mode="L")
    
    # Create the transparent image
    output_image = Image.new("RGBA", rgba_image.size, (0, 0, 0, 0))
    output_image.paste(rgba_image, mask=mask_image)
    
    return output_image

def crop_and_pad_object(image: Image.Image, padding_percentage: float = 0.1) -> Image.Image:
    """
    Crops a transparent PNG to its non-transparent bounds, then pads it to a square
    with a transparent background, matching input requirements for 3D generators.
    """
    # Get bounding box of non-transparent areas
    bbox = image.getbbox()
    if not bbox:
        return image
        
    cropped = image.crop(bbox)
    w, h = cropped.size
    
    # Pad to square
    max_dim = max(w, h)
    padding = int(max_dim * padding_percentage)
    new_dim = max_dim + 2 * padding
    
    square_img = Image.new("RGBA", (new_dim, new_dim), (0, 0, 0, 0))
    # Center the cropped image
    x_offset = (new_dim - w) // 2
    y_offset = (new_dim - h) // 2
    square_img.paste(cropped, (x_offset, y_offset))
    
    return square_img

def run_inpainting_pipeline(
    inpaint_pipe, 
    image: Image.Image, 
    mask: np.ndarray, 
    prompt: str = "clean background, high resolution, seamless texture"
) -> Image.Image:
    """
    Fills in the masked area of the image using a Stable Diffusion Inpainting model.
    """
    # Invert mask (since we want to fill in the foreground objects' holes)
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_uint8, mode="L")
    
    # Resize to multiples of 8 (required by SD models)
    w, h = image.size
    new_w = (w // 8) * 8
    new_h = (h // 8) * 8
    
    resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_mask = mask_image.resize((new_w, new_h), Image.Resampling.NEAREST)
    
    # Run inpainting
    with torch.inference_mode():
        inpainted_image = inpaint_pipe(
            prompt=prompt,
            image=resized_image,
            mask_image=resized_mask,
            num_inference_steps=25
        ).images[0]
        
    # Resize back to original dimensions
    return inpainted_image.resize((w, h), Image.Resampling.LANCZOS)
