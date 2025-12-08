#!/bin/bash
# ExecuTools Setup Script
#
# This script is a convenient wrapper around the Makefile.
# For more control, use the Makefile directly:
#   make help     - Show all available targets
#   make cpp      - Build C++ component only
#   make python   - Build Python component only
#   make clean    - Clean all artifacts

set -e  # Exit on error

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# Parse command line arguments
case "$1" in
    clean)
        echo "Cleaning build environment..."
        make clean
        exit 0
        ;;
    cpp)
        echo "Building C++ component only..."
        make cpp
        exit 0
        ;;
    python)
        echo "Building Python component only..."
        make python
        exit 0
        ;;
    test)
        echo "Running tests..."
        make test
        exit 0
        ;;
    help|-h|--help)
        make help
        exit 0
        ;;
    "")
        # No arguments - build everything
        echo "Building all components..."
        make all
        echo ""
        echo "Setup complete!"
        echo "To activate the Python environment: source .venv/bin/activate"
        exit 0
        ;;
    *)
        echo "Unknown option: $1"
        echo "Run './setup.sh help' for usage information"
        exit 1
        ;;
esac

