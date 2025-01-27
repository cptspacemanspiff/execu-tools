#pragma once

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/result.h>
#include <string>
#include <tokenizers_cpp.h>
#include <vector>

namespace executools {

class HFStringTokenizer {
public:
  HFStringTokenizer(const std::string &tokenizer_blob_str);
  // strings to tensors:
  executorch::runtime::Result<std::pair<executorch::extension::TensorPtr,
                                        executorch::extension::TensorPtr>>
  strings_to_tensors(const std::vector<std::string> &input_strings);
  // tensors to strings:
  std::vector<std::string>
  tensors_to_strings(const executorch::extension::TensorPtr &tensor_ptr,
                     bool skip_special_tokens = false);

private:
  std::unique_ptr<tokenizers::HFTokenizer> tokenizer_;
};

} // namespace executools