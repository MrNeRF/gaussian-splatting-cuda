#pragma once

#include <filesystem>
#include <torch/torch.h>
#include <tuple>
#include <vector>

namespace gs {

    class FrequencyScheduler {
    public:
        struct ResolutionStage {
            float factor; // Downscaling factor (e.g., 0.25 = 1/4 resolution)
            int steps;    // Number of steps at this resolution
        };

        FrequencyScheduler() = default;

        // Initialize from image paths
        void initialize(const std::vector<std::filesystem::path>& image_paths,
                        int total_iterations,
                        torch::Device device = torch::kCUDA);

        // Get current downscaling factor for given iteration
        float get_factor_for_iteration(int iteration) const;

        // Check if scheduler is enabled and initialized
        bool is_enabled() const { return !schedule_.empty(); }

        // Get the full schedule for logging
        const std::vector<ResolutionStage>& get_schedule() const { return schedule_; }

    private:
        std::vector<ResolutionStage> schedule_;
        std::vector<int> cumulative_steps_;

        // Compute frequency energy of an image
        static float compute_frequency_energy(const torch::Tensor& img);

        // Process a single image and return energies
        static std::tuple<float, std::vector<std::pair<float, float>>>
        process_image(const std::filesystem::path& path,
                      const std::vector<float>& candidate_factors,
                      torch::Device device);

        // Compute dataset-wide frequency metrics
        static std::tuple<float, std::vector<std::pair<float, float>>>
        compute_dataset_freq_metrics(const std::vector<std::filesystem::path>& image_paths,
                                     torch::Device device);

        // Allocate iterations based on frequency ratios
        static std::vector<ResolutionStage>
        allocate_iterations_by_frequency(int total_iterations,
                                         float full_energy,
                                         const std::vector<std::pair<float, float>>& downsampled_energies);
    };

} // namespace gs