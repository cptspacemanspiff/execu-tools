# setup test directory header files:
set(EXECUTOOLS_PYTHON_ARTIFACT_DIR
    ${CMAKE_CURRENT_LIST_DIR}/../../python/tests/export_artifacts)
configure_file(${CMAKE_CURRENT_LIST_DIR}/ExecuToolsTestDirs.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/ExecuToolsTestDirs.h)

# ##############################################################################
# Manual EXEs will be moved once more mature, faster to implement here.
# ##############################################################################

FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG 6c7b07a878ad834957b98d0f9ce1dbe0cb204fc9 # v2.4.2 May 4, 2024
  OVERRIDE_FIND_PACKAGE SYSTEM)

FetchContent_MakeAvailable(CLI11)

find_package(CLI11 CONFIG REQUIRED)

cmt_add_executable(ExecuToolsEncoderDecoder
                   manual_encoder_decoder_runner.cpp)

target_link_libraries(${CMT_LAST_TARGET} PRIVATE ExecuTools::ExecuTools
                                                etdump
                                                 CLI11::CLI11)

# ##############################################################################
# Manual Test stuff for initial debugging... Individual capabilities will be
# moved to Unit Tests, and these will be removed.
# ##############################################################################

cmt_add_executable(ModuleTest NAMESPACED test_executorch.cpp)
target_link_libraries(
  ${CMT_LAST_TARGET}
  PRIVATE executorch
          portable_ops_lib # not the best implementation, replace with backend
                           # specific.
          extension_data_loader
          extension_module_static
          extension_tensor
          ExecuTools::ExecuTools
          ExecuTools::ExecuToolsUtils
          etdump)

cmt_target_headers(
  ${CMT_LAST_TARGET} PRIVATE BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR} FILES
  ${CMAKE_CURRENT_BINARY_DIR}/ExecuToolsTestDirs.h)

cmt_add_executable(Manual NAMESPACED test_executorch_manual.cpp)

target_link_libraries(
  ${CMT_LAST_TARGET}
  PRIVATE executorch
          portable_ops_lib # not the best implementation, replace with backend
                           # specific.
          extension_data_loader
          extension_module_static
          extension_tensor
          ExecuTools::ExecuToolsUtils
          etdump)

cmt_target_headers(
  ${CMT_LAST_TARGET} PRIVATE BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR} FILES
  ${CMAKE_CURRENT_BINARY_DIR}/ExecuToolsTestDirs.h)

# ##############################################################################
# Actual Unit Tests
# ##############################################################################

cmt_add_executable(UnitTests NAMESPACED test_shared_memory_manager.cpp)
target_link_libraries(
  ${CMT_LAST_TARGET}
  PRIVATE executorch
          portable_ops_lib # not the best implementation, replace with backend
                           # specific.
          extension_data_loader
          extension_module_static
          extension_tensor
          ExecuTools::ExecuToolsUtils
          etdump
          GTest::gtest_main)

# target_compile_options(${CMT_LAST_TARGET} INTERFACE -DET_EVENT_TRACER_ENABLED)
include(GoogleTest)
gtest_discover_tests(${CMT_LAST_TARGET})
