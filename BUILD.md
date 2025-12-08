# ExecuTools Build System

This project has been restructured to use a Makefile-based build system that cleanly separates the C++ and Python components.

## Project Structure

```
execu-tools/
├── cpp/                    # C++ component
│   ├── src/               # C++ source files
│   ├── include/           # C++ headers
│   ├── test/              # C++ tests
│   └── CMakeLists.txt     # CMake configuration
├── python/                # Python component
│   ├── execu_tools/       # Python package
│   ├── tests/             # Python tests
│   └── pyproject.toml     # Python package config
├── executorch/            # Executorch dependency (git submodule)
├── transformers/          # Transformers fork (optional)
├── Makefile               # Main build system
└── setup.sh               # Convenience wrapper script
```

## Dependencies

### C++ Component
- **Build-time dependencies**: Python runtime with torch, torchgen, pyyaml (for code generation)
- **Library dependencies**: Executorch C++ runtime (fetched via CMake FetchContent), tokenizers-cpp, GoogleTest
- **Build system**: CMake + Make
- **Important**: The C++ build requires Python tooling because executorch uses Python scripts to generate C++ kernel registration code

### Python Component
- **Depends on**: Executorch Python package (which depends on the C++ runtime being built first)
- **Build system**: Poetry/uv
- **Package manager**: uv

## Building

### Quick Start

Build everything:
```bash
./setup.sh
# or
make
```

### Building Components Separately

Build only the C++ component:
```bash
./setup.sh cpp
# or
make cpp
```

Build only the Python component:
```bash
./setup.sh python
# or
make python
```

Note: The Python component automatically builds the Executorch Python package as a dependency, which itself depends on the Executorch C++ runtime.

### Available Make Targets

```bash
make help              # Show all available targets
make all               # Build both C++ and Python (default)
make cpp               # Build C++ component only
make python            # Build Python component only
make executorch-python # Build/install Executorch Python package
make venv              # Create Python virtual environment
make test              # Run Python tests
make clean             # Remove all build artifacts
make clean-cpp         # Remove C++ build artifacts only
make clean-python      # Remove Python artifacts only
```

### Build Order and Dependencies

The build system automatically handles dependencies in this order:

1. **Virtual Environment** (`make venv`):
   - Creates Python virtual environment in `.venv/`
   - Installs basic requirements from `requirements-dev.txt`

2. **Executorch Build Dependencies** (`make executorch-deps`):
   - Depends on: venv
   - Installs PyTorch and torchgen (needed for C++ code generation)
   - Runs `executorch/install_requirements.sh`

3. **C++ Component** (`make cpp`):
   - Depends on: executorch-deps
   - Creates build directory: `cpp/build/`
   - Runs CMake to configure (downloads dependencies via FetchContent)
   - Builds C++ libraries and executables
   - **Note**: Python must be activated during build for code generation scripts

4. **Executorch Python** (`make executorch-python`):
   - Depends on: cpp (C++ runtime must be built first)
   - Runs executorch install scripts
   - Installs executorch Python package

5. **Python Component** (`make python`):
   - Depends on: executorch-python
   - Installs execu-tools Python package

## Cleaning

Clean everything:
```bash
./setup.sh clean
# or
make clean
```

Clean only C++ artifacts:
```bash
make clean-cpp
```

Clean only Python artifacts (venv, etc):
```bash
make clean-python
```

## Testing

Run Python tests:
```bash
./setup.sh test
# or
make test
```

## Configuration

### Environment Variables

- `JOBS`: Number of parallel build jobs (default: 8)
  ```bash
  make cpp JOBS=16
  ```

### CMake Options

CMake options can be modified in [cpp/CMakeLists.txt](cpp/CMakeLists.txt), including:
- `EXECUTOOLS_LINK_OPTIMIZED`: Use optimized vs portable kernels
- Executorch build options (XNNPACK, event tracer, etc.)

## Virtual Environment

After building, activate the Python environment:
```bash
source .venv/bin/activate
```

## Troubleshooting

### Build fails with CMake errors
- Ensure CMake 3.27+ is installed
- Try cleaning: `make clean-cpp` then rebuild

### Python package not found
- Ensure the venv is activated: `source .venv/bin/activate`
- Rebuild Python component: `make python`

### Executorch build fails
- Check that the executorch submodule is initialized
- Ensure the executorch directory exists at the project root
