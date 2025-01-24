#include <ExecuTools/encoder_decoder_runner.h>

using namespace executools;

EncoderDecoderRunner::EncoderDecoderRunner(
    const std::string &model_path,
    executorch::extension::Module::LoadMode load_mode,
    std::unique_ptr<executorch::runtime::EventTracer> event_tracer)
    : MultiEntryPointRunner(model_path, load_mode, std::move(event_tracer)) {



    }



void EncoderDecoderRunner::initialize_tokenizer() {

    // pull the tokenizer json from the model:

}
