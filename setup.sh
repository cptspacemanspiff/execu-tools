#!/bin/bash
# if you are running this alot, gitcache is usefull, as it pulls the git repos 
# once and then caches them, allowing for clean builds quickly even if you are 
# running offline.

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

clean_environment() {
    echo "Cleaning build environment..."
    rm -rf cpp/build
    rm -rf .executools_venv
    echo "Clean completed"
}

# Parse command line arguments
if [ "$1" = "clean" ]; then
    clean_environment
    exit 0
fi

# Create and activate Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv .executools_venv
source .executools_venv/bin/activate

# Show which Python environment is being used
echo "Using Python environment: $(which python)"

# Regular setup continues
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu -U
pip install tomli zstd tomli

# download executorch:
unset CC CXX # make sure we use the system compiler
cd cpp && mkdir -p build && cd build
# we are at cpp/build
cmake .. -G Ninja # this downloads all c++ side deps
# ninja

# build executorch python package:
./_deps/executorch/install_requirements.sh

echo "Cloning transformers repo"
git clone git@github.com:cptspacemanspiff/transformers.git