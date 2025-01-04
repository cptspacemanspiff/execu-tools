cmt_add_executable(UnitTests NAMESPACED test_executorch.cpp)
target_link_libraries(${CMT_LAST_TARGET} PRIVATE
    executorch
    portable_ops_lib # not the best implementation, replace with backend specific.
    extension_data_loader
    extension_module_static
    extension_tensor
    ExecuTools::ExecuToolsUtils
    )