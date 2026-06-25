# 1. Clone the Hunyuan3D-2 repository
git clone https://github.com/Tencent/Hunyuan3D-2.git
cd Hunyuan3D-2

# 2. Install dependencies (including trimesh, einops, etc.)
pip install -r requirements.txt
pip install .

# 3. Go back to your project directory and run the preprocessor
cd ../world_gen
python world_preprocessor/main.py --image "../world.png" --output_dir "./output_scene"
