#pragma once

#include <memory>

namespace gs {

    namespace param {
        struct TrainingParameters;
    } // namespace param

    struct TrainingParameters;

    class Application {
    public:
        int run(std::unique_ptr<param::TrainingParameters> params);
    };

} // namespace gs
