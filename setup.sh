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
    rm -rf .venv
    echo "Clean completed"
}

# Parse command line arguments
if [ "$1" = "clean" ]; then
    clean_environment
    exit 0
fi

# Create and activate Python virtual environment
echo "Creating Python virtual environment..."
if [ ! -d ".venv" ] || [ $UV_VENV_CLEAR -eq 1 ]; then
    uv venv
fi
source .venv/bin/activate
uv pip install -r requirements-dev.txt

# Show which Python environment is being used
echo "Using Python environment: $(which python)"

# Regular setup continues
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu -U
# uv pip install tomli zstd tomli cmake==3.31.6 ninja

# download executorch:
unset CC CXX # make sure we use the system compiler
cd cpp && mkdir -p build && cd build
# we are at cpp/build
cmake .. # this downloads all c++ side deps
# ninja

# build + installexecutorch python package:
cd ../executorch/
./install_requirements.sh
./install_executorch.sh

# # install our updated versions of transformers:
cd ../../../../  # we are now at the root of the repo
# uv pip install -e transformers

# # install executools python component:
# uv pip install -e python

# We have built and installed the executorch python package, now build the c++ side of the project
cd cpp/build
make -j 8

# everything is built.
echo "Setup complete"
exit 0

# run the tests:
cd ../..
cd python
pytest

# generate the executorch models:

# echo "Cloning transformers repo"
# git clone git@github.com:cptspacemanspiff/transformers.git

# # Check if buck2 is running
# if ps aux | grep -v grep | grep "buck2" > /dev/null; then
#     echo "Error: buck2 process is running. Please stop it before running this script."
#     exit 1
# fi

