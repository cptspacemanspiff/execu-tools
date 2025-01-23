#include <gtest/gtest.h>

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

// #include <gtest/gtest.h>
// #include <gmock/gmock.h>
// #include <ExecuTools/shared_memory_manager.h>
// #include <executorch/runtime/executor/program.h>

// // Mock class for Program
// class MockProgram : public executorch::runtime::Program {
// public:
//     struct MethodConfig {
//         std::string name;
//         std::vector<size_t> buffer_sizes;
//     };

//     explicit MockProgram(std::vector<MethodConfig> methods) 
//         : methods_(std::move(methods)) {}

//     size_t num_methods() const override {
//         return methods_.size();
//     }

//     executorch::Error get_method_name(size_t index, const char** out) const override {
//         if (index >= methods_.size()) {
//             return executorch::Error::InvalidArgument;
//         }
//         *out = methods_[index].name.c_str();
//         return executorch::Error::Ok;
//     }

//     executorch::Error method_meta(const char* method_name, 
//                                 executorch::runtime::MethodMeta* out) const override {
//         for (const auto& method : methods_) {
//             if (method.name == method_name) {
//                 *out = createMethodMeta(method.buffer_sizes);
//                 return executorch::Error::Ok;
//             }
//         }
//         return executorch::Error::InvalidArgument;
//     }

// private:
//     executorch::runtime::MethodMeta createMethodMeta(
//         const std::vector<size_t>& buffer_sizes) const {
//         // Create a simple MethodMeta that only implements the memory planning interfaces
//         class MockMethodMeta : public executorch::runtime::MethodMeta {
//         public:
//             explicit MockMethodMeta(std::vector<size_t> sizes) 
//                 : buffer_sizes_(std::move(sizes)) {}

//             size_t num_memory_planned_buffers() const override {
//                 return buffer_sizes_.size();
//             }

//             executorch::Result<int64_t> memory_planned_buffer_size(
//                 size_t buffer_idx) const override {
//                 if (buffer_idx >= buffer_sizes_.size()) {
//                     return executorch::Error::InvalidArgument;
//                 }
//                 return buffer_sizes_[buffer_idx];
//             }

//         private:
//             std::vector<size_t> buffer_sizes_;
//         };

//         return MockMethodMeta(buffer_sizes);
//     }

//     std::vector<MethodConfig> methods_;
// };

// // Helper function to create a mock program
// std::shared_ptr<executorch::runtime::Program> createMockProgram(
//     const std::vector<std::pair<std::string, std::vector<size_t>>>& method_configs) {
//     std::vector<MockProgram::MethodConfig> configs;
//     for (const auto& config : method_configs) {
//         configs.push_back({config.first, config.second});
//     }
//     return std::make_shared<MockProgram>(std::move(configs));
// }