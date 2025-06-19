#include "core/frequency_scheduler.hpp"
#include "core/image_io.hpp"
#include <algorithm>
#include <execution>
#include <iostream>
#include <mutex>
#include <numeric>
#include <torch/torch.h>

namespace F = torch::nn::functional;

namespace gs {

    float FrequencyScheduler::compute_frequency_energy(const torch::Tensor& img) {
        // Expects [C, H, W] tensor
        if (img.dim() != 3) {
            throw std::runtime_error("Image must be [C, H, W]");
        }

        // Convert to grayscale if multi-channel
        torch::Tensor x = img.to(torch::kFloat32);
        if (img.size(0) > 1) {
            x = x.mean(0, /*keepdim=*/true); // [1, H, W]
        }

        // Compute 2D FFT
        auto fft_img = torch::fft::fft2(x, /*s=*/{}, /*dim=*/{-2, -1}, /*norm=*/"ortho");

        // Compute magnitude squared
        // For complex tensors, we need to use view_as_real to access real and imaginary parts
        auto fft_real_imag = torch::view_as_real(fft_img);
        auto real_part = fft_real_imag.select(-1, 0);
        auto imag_part = fft_real_imag.select(-1, 1);
        auto mag_sqr = real_part.square() + imag_part.square();

        return mag_sqr.sum().item<float>();
    }

    std::tuple<float, std::vector<std::pair<float, float>>>
    FrequencyScheduler::process_image(const std::filesystem::path& path,
                                      const std::vector<float>& candidate_factors,
                                      torch::Device device) {
        try {
            // Load image
            auto [data, w, h, c] = load_image(path);

            // Convert to tensor [H, W, C] -> [C, H, W]
            auto img_tensor = torch::from_blob(data, {h, w, c}, torch::kUInt8)
                                  .to(torch::kFloat32)
                                  .permute({2, 0, 1}) /
                              255.0f;
            img_tensor = img_tensor.to(device);

            // Compute full resolution energy
            float full_energy = compute_frequency_energy(img_tensor);

            // Compute downsampled energies
            std::vector<std::pair<float, float>> factor_energies;

            for (float factor : candidate_factors) {
                int new_h = static_cast<int>(h * factor);
                int new_w = static_cast<int>(w * factor);

                if (new_h < 2 || new_w < 2)
                    continue;

                // Downsample using area interpolation
                auto downsampled = F::interpolate(
                                       img_tensor.unsqueeze(0),
                                       F::InterpolateFuncOptions()
                                           .size(std::vector<int64_t>{new_h, new_w})
                                           .mode(torch::kArea))
                                       .squeeze(0);

                float energy = compute_frequency_energy(downsampled);
                factor_energies.push_back({factor, energy});
            }

            free_image(data);
            return {full_energy, factor_energies};

        } catch (const std::exception& e) {
            std::cerr << "Error processing " << path << ": " << e.what() << std::endl;
            return {0.0f, {}};
        }
    }

    std::tuple<float, std::vector<std::pair<float, float>>>
    FrequencyScheduler::compute_dataset_freq_metrics(const std::vector<std::filesystem::path>& image_paths,
                                                     torch::Device device) {
        if (image_paths.empty()) {
            throw std::runtime_error("No image paths provided");
        }

        const std::vector<float> candidate_factors = {0.2f, 0.25f, 0.333f, 0.5f};

        // Use regular floats with mutex for thread safety instead of atomics
        float full_energy_sum = 0.0f;
        int valid_count = 0;

        std::map<float, float> factor_sums;
        std::map<float, int> factor_counts;
        for (float f : candidate_factors) {
            factor_sums[f] = 0.0f;
            factor_counts[f] = 0;
        }

        // Mutex for thread-safe accumulation
        std::mutex acc_mutex;

        // Process images in parallel
        std::cout << "Analyzing frequency content of " << image_paths.size() << " images..." << std::endl;

        // Create indices for parallel processing
        std::vector<size_t> indices(image_paths.size());
        std::iota(indices.begin(), indices.end(), 0);

        // Process with progress reporting
        std::atomic<size_t> processed{0};

        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
                      [&](size_t idx) {
                          auto [full_e, factor_energies] = process_image(image_paths[idx], candidate_factors, device);

                          if (full_e > 0) {
                              std::lock_guard<std::mutex> lock(acc_mutex);
                              full_energy_sum += full_e;
                              valid_count++;

                              for (const auto& [factor, energy] : factor_energies) {
                                  factor_sums[factor] += energy;
                                  factor_counts[factor]++;
                              }
                          }

                          // Progress reporting
                          size_t current = ++processed;
                          if (current % 50 == 0 || current == image_paths.size()) {
                              std::cout << "\rFrequency analysis: " << current << "/" << image_paths.size() << std::flush;
                          }
                      });

        std::cout << std::endl;

        if (valid_count == 0) {
            throw std::runtime_error("Could not compute frequency energy for any image");
        }

        // Compute averages
        float avg_full = full_energy_sum / valid_count;

        std::vector<std::pair<float, float>> results;
        for (float f : candidate_factors) {
            if (factor_counts[f] > 0) {
                float avg_energy = factor_sums[f] / factor_counts[f];
                results.push_back({f, avg_energy});
            }
        }

        // Sort by factor
        std::sort(results.begin(), results.end());

        return {avg_full, results};
    }

    std::vector<FrequencyScheduler::ResolutionStage>
    FrequencyScheduler::allocate_iterations_by_frequency(int total_iterations,
                                                         float full_energy,
                                                         const std::vector<std::pair<float, float>>& downsampled_energies) {
        std::vector<ResolutionStage> schedule;
        int used_steps = 0;

        // Avoid division by zero
        if (full_energy <= 1e-9f) {
            std::cout << "Warning: Full frequency energy near zero. Using equal allocation." << std::endl;
            int num_stages = downsampled_energies.size() + 1;
            int steps_per_stage = total_iterations / num_stages;

            for (const auto& [factor, _] : downsampled_energies) {
                schedule.push_back({factor, steps_per_stage});
                used_steps += steps_per_stage;
            }

            int leftover = total_iterations - used_steps;
            schedule.push_back({1.0f, leftover});
            return schedule;
        }

        // Allocate based on frequency ratios
        for (const auto& [factor, energy] : downsampled_energies) {
            float fraction = std::max(0.0f, energy / full_energy);
            int steps = static_cast<int>(total_iterations * fraction);

            if (steps > 0) {
                schedule.push_back({factor, steps});
                used_steps += steps;
            }
        }

        // Allocate remaining steps to full resolution
        int leftover = total_iterations - used_steps;
        if (leftover > 0) {
            schedule.push_back({1.0f, leftover});
        } else if (schedule.empty() || schedule.back().factor != 1.0f) {
            // Ensure at least one step at full resolution
            if (!schedule.empty() && schedule.back().steps > 1) {
                schedule.back().steps--;
                schedule.push_back({1.0f, 1});
            } else {
                schedule.push_back({1.0f, 1});
            }
        }

        // Adjust if total doesn't match due to rounding
        int current_total = 0;
        for (const auto& stage : schedule) {
            current_total += stage.steps;
        }

        if (current_total != total_iterations && !schedule.empty()) {
            int diff = total_iterations - current_total;
            // Find the stage with most steps (usually full res)
            auto max_it = std::max_element(schedule.begin(), schedule.end(),
                                           [](const ResolutionStage& a, const ResolutionStage& b) {
                                               return a.steps < b.steps;
                                           });

            if (max_it != schedule.end()) {
                max_it->steps = std::max(1, max_it->steps + diff);
            }
        }

        return schedule;
    }

    void FrequencyScheduler::initialize(const std::vector<std::filesystem::path>& image_paths,
                                        int total_iterations,
                                        torch::Device device) {
        // Compute frequency metrics
        auto [full_energy, downsampled_energies] = compute_dataset_freq_metrics(image_paths, device);

        // Allocate iterations
        schedule_ = allocate_iterations_by_frequency(total_iterations, full_energy, downsampled_energies);

        // Build cumulative steps for efficient lookup
        cumulative_steps_.clear();
        int cumulative = 0;
        for (const auto& stage : schedule_) {
            cumulative += stage.steps;
            cumulative_steps_.push_back(cumulative);
        }

        // Log the schedule
        std::cout << "Generated Resolution Schedule (factor, steps):" << std::endl;
        for (const auto& stage : schedule_) {
            std::cout << "  " << stage.factor << "x resolution: " << stage.steps << " steps" << std::endl;
        }

        // Verify total
        int total = 0;
        for (const auto& stage : schedule_) {
            total += stage.steps;
        }
        if (total != total_iterations) {
            std::cout << "Warning: Schedule steps sum to " << total
                      << ", expected " << total_iterations << std::endl;
        }
    }

    float FrequencyScheduler::get_factor_for_iteration(int iteration) const {
        if (schedule_.empty()) {
            return 1.0f; // Full resolution if not initialized
        }

        // Find which stage we're in
        for (size_t i = 0; i < cumulative_steps_.size(); ++i) {
            if (iteration <= cumulative_steps_[i]) {
                return schedule_[i].factor;
            }
        }

        // Default to full resolution if beyond schedule
        return 1.0f;
    }

} // namespace gs