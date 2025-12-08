# ExecuTools Architecture

## Build Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                         make all                                 │
│                            │                                     │
│              ┌─────────────┴─────────────┐                       │
│              ▼                           ▼                       │
│           make cpp                   make python                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Detailed Dependency Tree                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  make python                                                     │
│      │                                                           │
│      └──▶ executorch-python                                      │
│              │                                                   │
│              └──▶ make cpp ──────────┐                           │
│                      │               │                           │
│                      └──▶ executorch-deps                         │
│                              │                                   │
│                              └──▶ make venv                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. Virtual Environment (`make venv`)
**Purpose**: Sets up Python environment with basic build tools

**Actions**:
- Creates `.venv/` directory
- Installs packages from `requirements-dev.txt`:
  - cmake, pip, pyyaml, setuptools, wheel, zstd
  - torch (for code generation)

**Output**: `.venv/bin/activate`

---

### 2. Executorch Dependencies (`make executorch-deps`)
**Purpose**: Installs Python packages required for C++ code generation

**Dependencies**: venv

**Actions**:
- Runs `executorch/install_requirements.sh`
- Installs PyTorch with torchgen
- Installs additional executorch Python dependencies

**Output**: `.venv/executorch-deps-installed` (marker file)

**Why needed**: Executorch's C++ build runs Python scripts that:
- Parse YAML operator definitions
- Generate C++ kernel registration code
- Requires `torchgen` package from PyTorch

---

### 3. C++ Component (`make cpp`)
**Purpose**: Builds the C++ runtime and libraries

**Dependencies**: executorch-deps

**Actions**:
1. **CMake Configuration** (`cpp/build/Makefile` target):
   - Creates `cpp/build/` directory
   - Runs CMake with Python environment activated
   - Downloads dependencies via FetchContent:
     - executorch C++ runtime (from `../executorch/`)
     - tokenizers-cpp
     - GoogleTest
   - Configures build options (XNNPACK, optimized kernels, etc.)

2. **Build** (`make cpp` target):
   - Compiles C++ sources with Python activated
   - Runs executorch codegen scripts during build:
     - `executorch/codegen/gen.py` - generates kernel registration
     - Creates `RegisterCodegenUnboxedKernelsEverything.cpp`
   - Links libraries

**Output**:
- `cpp/build/` directory with compiled libraries
- Executorch C++ runtime libraries
- ExecuTools C++ libraries and executables

**Key Insight**: Even though this is the "C++ component", it requires Python at **build time** for code generation, not just at runtime.

---

### 4. Executorch Python Package (`make executorch-python`)
**Purpose**: Builds and installs the executorch Python package

**Dependencies**: cpp (must be built first)

**Actions**:
- Runs `executorch/install_executorch.sh`
- Builds Python extension modules that link against C++ runtime
- Installs executorch Python package into venv

**Output**: executorch package installed in `.venv/lib/python*/site-packages/`

**Why depends on C++**: The Python package contains extension modules that need to link against the compiled C++ runtime libraries.

---

### 5. Python Component (`make python`)
**Purpose**: Installs the execu-tools Python package

**Dependencies**: executorch-python

**Actions**:
- Runs `uv pip install -e python/`
- Installs execu-tools package in editable mode

**Output**: execu-tools package installed in `.venv/lib/python*/site-packages/`

---

## Why C++ Needs Python

The C++ build has an unusual dependency on Python because:

1. **Code Generation**: Executorch uses a Python-based code generation system
   - Operator definitions are in YAML files
   - Python scripts parse YAML and generate C++ code
   - This happens **during** the C++ build, not before it

2. **Required Python Packages**:
   - `torch` - provides `torchgen` for operator parsing
   - `pyyaml` - for reading operator definitions
   - Various executorch Python modules used by codegen scripts

3. **Build Process**:
   ```
   CMake → triggers custom command → runs Python script → generates .cpp → compiles C++
   ```

This is why you can't truly build the C++ component in isolation - it needs Python tooling at build time.

## Building Components Separately

### To build only C++ runtime (but you still need Python):
```bash
make cpp
# This will:
# 1. Create venv
# 2. Install executorch Python deps (torch, etc.)
# 3. Configure and build C++
```

### To build only Python package (will build C++ first):
```bash
make python
# This will:
# 1. Create venv
# 2. Install executorch Python deps
# 3. Build C++ runtime
# 4. Install executorch Python package
# 5. Install execu-tools Python package
```

## Clean Targets

- `make clean` - Remove everything (C++ and Python)
- `make clean-cpp` - Remove only `cpp/build/`
- `make clean-python` - Remove only `.venv/` and Python artifacts

Note: Cleaning Python artifacts will require reinstalling executorch dependencies before the C++ can build again.
