#pragma once

#include "core/dataset.hpp"
#include "core/parameters.hpp"
#include "core/rasterizer.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

class splatData;
namespace gs {
    namespace metrics {

        // SSIM window creation
        torch::Tensor gaussian(const int window_size, const float sigma);
        torch::Tensor create_window(const int window_size, const int channel);

        // Peak Signal-to-Noise Ratio
        class PSNR {
        public:
            explicit PSNR(const float data_range = 1.0f) : data_range_(data_range) {}

            float compute(const torch::Tensor& pred, const torch::Tensor& target) const;

        private:
            const float data_range_;
        };

        // Structural Similarity Index
        class SSIM {
        public:
            SSIM(const int window_size = 11, const int channel = 3);

            float compute(const torch::Tensor& pred, const torch::Tensor& target);

        private:
            const int window_size_;
            const int channel_;
            torch::Tensor window_;
            static constexpr float C1 = 0.01f * 0.01f;
            static constexpr float C2 = 0.03f * 0.03f;
        };

        class LPIPS {
        public:
            explicit LPIPS(const std::string& model_path = "");

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
            explicit MetricsReporter(const std::filesystem::path& output_dir);

            void add_metrics(const EvalMetrics& metrics);
            void save_report() const;

        private:
            const std::filesystem::path output_dir_;
            std::vector<EvalMetrics> all_metrics_;
            const std::filesystem::path csv_path_;
            const std::filesystem::path txt_path_;
        };

        // Main evaluator class that handles all metrics computation and visualization
        class MetricsEvaluator {
        public:
            explicit MetricsEvaluator(const param::TrainingParameters& params);

            // Check if evaluation is enabled
            bool is_enabled() const { return _params.optimization.enable_eval; }

            // Check if we should evaluate at this iteration
            bool should_evaluate(const int iteration) const;

            // Main evaluation method
            EvalMetrics evaluate(const int iteration,
                                 const SplatData& splatData,
                                 std::shared_ptr<CameraDataset> val_dataset,
                                 torch::Tensor& background);

            // Save final report
            void save_report() const {
                if (_reporter)
                    _reporter->save_report();
            }

            // Print evaluation header
            void print_evaluation_header(const int iteration) const {
                std::cout << std::endl;
                std::cout << "[Evaluation at step " << iteration << "]" << std::endl;
            }

        private:
            // Configuration
            const param::TrainingParameters& _params;

            // Metrics
            std::unique_ptr<PSNR> _psnr_metric;
            std::unique_ptr<SSIM> _ssim_metric;
            std::unique_ptr<LPIPS> _lpips_metric;
            std::unique_ptr<MetricsReporter> _reporter;

            // Helper functions
            torch::Tensor apply_depth_colormap(const torch::Tensor& depth_normalized) const;
            bool has_rgb() const;
            bool has_depth() const;

            // Create dataloader from dataset
            auto make_dataloader(std::shared_ptr<CameraDataset> dataset, const int workers = 1) const;
        };

    } // namespace metrics
} // namespace gs