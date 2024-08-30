//
// Created by paja on 8/30/24.
//

#ifndef GAUSSIAN_SPLATTING_CUDA_ARGS_PARSER_CUH
#define GAUSSIAN_SPLATTING_CUDA_ARGS_PARSER_CUH

namespace gs {
    namespace param {
        struct OptimizationParameters;
        struct ModelParameters;
    }
}

class ArgsParser {
public:
    static int Parse(int argc, const char* argv[],
                            gs::param::ModelParameters& modelParams,
                            gs::param::OptimizationParameters& optimParams);

    static int Dump(const gs::param::ModelParameters& params);
};


#endif // GAUSSIAN_SPLATTING_CUDA_ARGS_PARSER_CUH
