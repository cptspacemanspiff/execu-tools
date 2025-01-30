cmake_minimum_required(VERSION 3.27)
project(ExecuTools VERSION 0.0.1)
# Get shared cmake stuff:
include(FetchContent)
FetchContent_Declare(
  cmaketools
  GIT_REPOSITORY git@github.com:cptspacemanspiff/CMakeTools.git
  GIT_TAG main)

FetchContent_MakeAvailable(cmaketools)

set(CMAKE_CXX_STANDARD 20)

cmt_project_setup()

add_compile_definitions(ET_EVENT_TRACER_ENABLED=1)

# args for executorch (not best place for this):
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON)
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON)
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON)
set(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL ON)
set(EXECUTORCH_BUILD_DEVTOOLS ON)
set(EXECUTORCH_BUILD_XNNPACK ON)

FetchContent_Declare(
  executorch
  GIT_REPOSITORY https://github.com/cptspacemanspiff/executorch.git
  GIT_TAG multiple-entrypoint-shared-state
  # root of executorch must be named executorch see:
  # https://github.com/pytorch/executorch/issues/6475
  SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/executorch" OVERRIDE_FIND_PACKAGE
  SYSTEM)
FetchContent_MakeAvailable(executorch)

FetchContent_Declare(
  tokenizers-cpp
  # GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git GIT_TAG main
  GIT_REPOSITORY https://github.com/cptspacemanspiff/tokenizers-cpp.git
  GIT_TAG return-attn-mask-from-hftokenizer
  OVERRIDE_FIND_PACKAGE SYSTEM)
FetchContent_MakeAvailable(tokenizers-cpp)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG b514bdc898e2951020cbdca1304b75f5950d1f59 # release-1.15.2
  OVERRIDE_FIND_PACKAGE SYSTEM)

FetchContent_MakeAvailable(googletest)

# Setup executorch library linking targets:
if(EXECUTOOLS_LINK_OPTIMIZED)
  message(STATUS "ExecuTools: Linking optimized libraries")
  set(EXECUTOOLS_LINK_LIBS optimized_native_cpu_ops_lib optimized_kernels
                           portable_kernels)
else()
  message(STATUS "ExecuTools: Linking portable libraries")
  set(EXECUTOOLS_LINK_LIBS portable_ops_lib portable_kernels)
endif()

set(ET_LINK_LIBS_)
# cmt_add_doc_subdirectory(doc)
cmt_add_src_subdirectory(src)
cmt_add_test_subdirectory(test)
