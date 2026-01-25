.PHONY: all clean cpp python executorch-python venv help

# Configuration
PYTHON := python3
VENV_DIR := .venv
CPP_BUILD_DIR := cpp/build
EXECUTORCH_DIR := executorch
PYTHON_DIR := python
TRANSFORMERS_DIR := transformers
JOBS := 8

# Default target
all: cpp python

help:
	@echo "ExecuTools Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all                - Build both C++ and Python components (default)"
	@echo "  cpp                - Build C++ component only"
	@echo "  python             - Build Python component only (depends on executorch-python)"
	@echo "  executorch-deps    - Install Python dependencies needed for C++ build"
	@echo "  executorch-python  - Build and install executorch Python package"
	@echo "  venv               - Create Python virtual environment"
	@echo "  clean              - Remove all build artifacts"
	@echo "  clean-cpp          - Remove C++ build artifacts only"
	@echo "  clean-python       - Remove Python artifacts only"
	@echo "  test               - Run Python tests"
	@echo ""
	@echo "Build Dependencies:"
	@echo "  cpp -> executorch-deps -> venv"
	@echo "  executorch-python -> cpp"
	@echo "  python -> executorch-python"
	@echo ""
	@echo "Note: C++ build requires Python runtime (torch, torchgen) for code generation"
	@echo ""
	@echo "Usage:"
	@echo "  make              # Build everything"
	@echo "  make cpp          # Build only C++ component (installs Python deps first)"
	@echo "  make python       # Build only Python component (builds C++ first)"
	@echo "  make clean        # Clean everything"

# Virtual environment setup
venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: requirements-dev.txt
	@echo "Creating Python virtual environment..."
	@uv venv $(VENV_DIR) --python 3.12
	@. $(VENV_DIR)/bin/activate && uv pip install -r requirements-dev.txt
	@touch $(VENV_DIR)/bin/activate

# Executorch build-time dependencies (needed for C++ code generation)
# This installs torch and torchgen which are required by executorch's codegen
executorch-deps: venv
	@echo "Installing executorch build-time dependencies..."
	@. $(VENV_DIR)/bin/activate && \
		cd $(EXECUTORCH_DIR) && \
		./install_requirements.sh
	@touch $(VENV_DIR)/executorch-deps-installed

# C++ component (depends on Python runtime for code generation)
# Note: Executorch C++ build requires Python packages (torchgen, etc) for code generation
cpp: executorch-deps $(CPP_BUILD_DIR)/Makefile
	@echo "Building C++ component..."
	@. $(VENV_DIR)/bin/activate && $(MAKE) -C $(CPP_BUILD_DIR) -j $(JOBS)
	@echo "C++ build complete"

$(CPP_BUILD_DIR)/Makefile: executorch-deps
	@echo "Configuring C++ build with CMake..."
	@mkdir -p $(CPP_BUILD_DIR)
	@. $(VENV_DIR)/bin/activate && cd $(CPP_BUILD_DIR) && cmake ..

# Executorch Python package (depends on C++ runtime being built)
executorch-python: cpp
	@echo "Building and installing executorch Python package..."
	@. $(VENV_DIR)/bin/activate && \
		cd $(EXECUTORCH_DIR) && \
		./install_executorch.sh
	@echo "Executorch Python package installed"

# Python component (depends on executorch Python package)
python: executorch-python
	@echo "Installing ExecuTools Python package..."
	@. $(VENV_DIR)/bin/activate && \
		uv pip install -e $(PYTHON_DIR)
	@echo "Python component installed"

# Optional: Install transformers fork if needed
transformers: venv
	@echo "Installing transformers fork..."
	@. $(VENV_DIR)/bin/activate && \
		uv pip install -e $(TRANSFORMERS_DIR)

# Testing
test: python
	@echo "Running Python tests..."
	@. $(VENV_DIR)/bin/activate && \
		cd $(PYTHON_DIR) && \
		pytest

# Cleaning targets
clean: clean-cpp clean-python

clean-cpp:
	@echo "Cleaning C++ build artifacts..."
	@rm -rf $(CPP_BUILD_DIR)
	@echo "C++ clean complete"

clean-python:
	@echo "Cleaning Python artifacts..."
	@rm -rf $(VENV_DIR)
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Python clean complete"
