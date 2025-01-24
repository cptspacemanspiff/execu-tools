
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <ExecuTools/multi_entry_point_runner.h>

namespace executools {



class EncoderDecoderRunner : public MultiEntryPointRunner {

public:
  EncoderDecoderRunner(const std::string &model_path);

protected:
};

} // namespace executools