# library that helps with loggings and strifying some of the types.
cmt_add_library(ExecuTools)

target_sources(
  ExecuTools
  PRIVATE shared_memory_manager.cpp encoder_decoder_runner.cpp
          multi_entry_point_runner.cpp reserved_fn_names.cpp
          constant_method_utils.cpp)
target_link_libraries(
  ExecuTools PUBLIC executorch extension_data_loader extension_module_static
                    extension_tensor tokenizers_cpp portable_ops_lib)

cmt_target_headers(
  ExecuTools PUBLIC BASE_DIRS ${PROJECT_SOURCE_DIR}/include FILES
  ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/shared_memory_manager.h)

add_subdirectory(utils)
add_subdirectory(executables)
