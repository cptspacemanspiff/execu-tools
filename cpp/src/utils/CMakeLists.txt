# library that helps with loggings and strifying some of the types.
cmt_add_library(ExecuToolsUtils)

target_sources(ExecuToolsUtils PRIVATE string_helpers.cpp)
target_link_libraries(ExecuToolsUtils PUBLIC executorch)

cmt_target_headers(ExecuToolsUtils
  PUBLIC
  BASE_DIRS ${PROJECT_SOURCE_DIR}/include
  FILES ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/utils/string_helpers.h)