# setup test directory header files:
set(EXECUTOOLS_PYTHON_ARTIFACT_DIR
    ${CMAKE_CURRENT_LIST_DIR}/../../python/tests/export_artifacts)
configure_file(${CMAKE_CURRENT_LIST_DIR}/ExecuToolsTestDirs.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/ExecuToolsTestDirs.h)

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
          ExecuTools::ExecuToolsUtils)

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

################################################################################
#Actual Unit Tests
################################################################################

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