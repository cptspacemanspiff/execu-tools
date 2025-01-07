# library that helps with loggings and strifying some of the types.
cmt_add_library(ExecuTools)

target_sources(ExecuTools PRIVATE shared_memory_manager.cpp)
target_link_libraries(ExecuTools PUBLIC executorch)

cmt_target_headers(
  ExecuTools PUBLIC BASE_DIRS ${PROJECT_SOURCE_DIR}/include FILES
  ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/shared_memory_manager.h)

add_subdirectory(utils)
