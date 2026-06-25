import os
import torch

# General Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./output_scene"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. SAM 2 Configuration
# Using HF transformers Sam2Model
SAM2_MODEL_ID = "facebook/sam2-hiera-small" 
SAM2_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# 2. Depth Anything V2 Configuration
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# 3. Inpainting Configuration
INPAINT_MODEL_ID = "runwayml/stable-diffusion-inpainting"
INPAINT_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# 4. 3D Mesh Generation Configuration
# We support hunyuan3d, stablefast3d, or triposr
MESH_GENERATOR_TYPE = "hunyuan3d"  
HUNYUAN_MODEL_ID = "tencent/Hunyuan3D-2mini"
HUNYUAN_SUBFOLDER = "hunyuan3d-dit-v2-mini"
SF3D_MODEL_ID = "stabilityai/stablefast3d"
TRIPOSR_MODEL_ID = "stabilityai/TripoSR"

# Optimization flags
ENABLE_CPU_OFFLOAD = True
ENABLE_ATTENTION_SLICING = True

# 5. VLM Configuration
VLM_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
VLM_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
