#pragma once

#define DEF inline constexpr

namespace fast_gs::optimizer::config {
    DEF bool debug = false;
    // block size constants
    DEF int block_size_adam_step = 256;
}

namespace config = fast_gs::optimizer::config;

#undef DEF
