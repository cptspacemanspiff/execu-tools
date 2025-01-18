pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu -U
pip install tomli zstd yaml

# download executorch:
cd cpp && mkdir -p build && cd build 
cmake .. -G Ninja 
# ninja

# build executorch:


# git clone git@github.com:cptspacemanspiff/transformers.git