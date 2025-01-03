cmake_minimum_required(VERSION 3.27)
project(ExecuTools VERSION 0.0.1)
# Get shared cmake stuff:
include(FetchContent)
FetchContent_Declare(
  cmaketools
  GIT_REPOSITORY git@github.com:cptspacemanspiff/CMakeTools.git
  GIT_TAG        main
)

FetchContent_MakeAvailable(cmaketools)


set(CMAKE_CXX_STANDARD 20)

cmt_project_setup()

# args for executorch:
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON )
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON )
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON)
set(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL ON)

FetchContent_Declare(
    executorch
    GIT_REPOSITORY git@github.com:pytorch/executorch.git
    GIT_TAG        v0.4.0
    # root of executorch must be named executorch see:
    # https://github.com/pytorch/executorch/issues/6475
    SOURCE_DIR     "${CMAKE_BINARY_DIR}/_deps/executorch" 
    
    OVERRIDE_FIND_PACKAGE 
    SYSTEM
)

FetchContent_MakeAvailable(executorch)

# cmt_add_doc_subdirectory(doc)
cmt_add_src_subdirectory(src)
cmt_add_test_subdirectory(test)
