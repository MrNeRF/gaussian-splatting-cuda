#pragma once

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

namespace gs {
    namespace metrics {

        // SSIM window creation
        torch::Tensor gaussian(int window_size, float sigma);
        torch::Tensor create_window(int window_size, int channel);

        // Peak Signal-to-Noise Ratio
        class PSNR {
        public:
            PSNR(float data_range = 1.0f) : data_range_(data_range) {}

            float compute(const torch::Tensor& pred, const torch::Tensor& target);

        private:
            float data_range_;
        };

        // Structural Similarity Index
        class SSIM {
        public:
            SSIM(int window_size = 11, int channel = 3);

            float compute(const torch::Tensor& pred, const torch::Tensor& target);

        private:
            int window_size_;
            int channel_;
            torch::Tensor window_;
            static constexpr float C1 = 0.01f * 0.01f;
            static constexpr float C2 = 0.03f * 0.03f;
        };

        class LPIPS {
        public:
            LPIPS(const std::string& model_path = "");

            float compute(const torch::Tensor& pred, const torch::Tensor& target);
            bool is_loaded() const { return model_loaded_; }

        private:
            torch::jit::script::Module model_;
            bool model_loaded_ = false;

            void load_model(const std::string& model_path);
        };

        // Evaluation result structure
        struct EvalMetrics {
            float psnr;
            float ssim;
            float lpips;
            float elapsed_time;
            int num_gaussians;
            int iteration;

            std::string to_string() const {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(4);
                ss << "PSNR: " << psnr
                   << ", SSIM: " << ssim
                   << ", LPIPS: " << lpips
                   << ", Time: " << elapsed_time << "s/image"
                   << ", #GS: " << num_gaussians;
                return ss.str();
            }

            std::string to_csv_header() const {
                return "iteration,psnr,ssim,lpips,time_per_image,num_gaussians";
            }

            std::string to_csv_row() const {
                std::stringstream ss;
                ss << iteration << ","
                   << std::fixed << std::setprecision(6)
                   << psnr << ","
                   << ssim << ","
                   << lpips << ","
                   << elapsed_time << ","
                   << num_gaussians;
                return ss.str();
            }
        };

        // Metrics reporter class
        class MetricsReporter {
        public:
            MetricsReporter(const std::filesystem::path& output_dir);

            void add_metrics(const EvalMetrics& metrics);
            void save_report();

        private:
            std::filesystem::path output_dir_;
            std::vector<EvalMetrics> all_metrics_;
            std::filesystem::path csv_path_;
            std::filesystem::path txt_path_;
        };

    } // namespace metrics
} // namespace gs