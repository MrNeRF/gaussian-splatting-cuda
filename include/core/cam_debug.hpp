#pragma once
#include <torch/torch.h>
#include <filesystem>
#include <fstream>
#include <iomanip>

namespace camdbg
{
    // create dir if needed
    inline void ensure_dir(const std::string& tag)
    {
        std::filesystem::create_directories("debug/" + tag);
    }

    // dump any tensor (float32/64, any shape) – one value per line
    inline void dump_tensor(const torch::Tensor& t,
                            const std::string& stem,
                            const std::string& tag)
    {
        ensure_dir(tag);
        std::ofstream f("debug/" + tag + "/" + stem + ".txt");
        f << std::setprecision(10);

        auto cpu = t.to(torch::kCPU).contiguous();
        if (cpu.dtype() == torch::kFloat64) cpu = cpu.to(torch::kFloat32);

        const float* p = cpu.data_ptr<float>();
        for (int64_t i = 0; i < cpu.numel(); ++i) f << p[i] << '\n';
    }

    // dump a single float (FoVx / FoVy etc.)
    inline void dump_scalar(float v,
                            const std::string& stem,
                            const std::string& tag)
    {
        ensure_dir(tag);
        std::ofstream("debug/" + tag + "/" + stem + ".txt")
            << std::setprecision(10) << v << '\n';
    }
}  // namespace camdbg
