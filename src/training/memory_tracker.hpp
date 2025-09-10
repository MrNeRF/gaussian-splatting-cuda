/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// memory_tracker.hpp
#pragma once

#include "core/logger.hpp"
#include "optimizers/fused_adam.hpp"
#include <c10/cuda/CUDACachingAllocator.h>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace gs::training {

    struct MemorySnapshot {
        size_t cuda_allocated_bytes;
        size_t cuda_reserved_bytes;
        size_t cuda_active_bytes;
        size_t cuda_inactive_bytes; // We'll calculate this as reserved - allocated
        size_t system_free_bytes;
        size_t system_total_bytes;
        int iteration;
        std::string phase;
        std::chrono::steady_clock::time_point timestamp;
    };

    class MemoryTracker {
    public:
        static MemoryTracker& get() {
            static MemoryTracker instance;
            return instance;
        }

        void enable(bool detailed = false) {
            enabled_ = true;
            detailed_ = detailed;
            snapshots_.clear();
            start_time_ = std::chrono::steady_clock::now();
        }

        void disable() {
            enabled_ = false;
        }

        MemorySnapshot capture(int iteration, const std::string& phase) {
            MemorySnapshot snapshot;
            snapshot.iteration = iteration;
            snapshot.phase = phase;
            snapshot.timestamp = std::chrono::steady_clock::now();

            // CUDA memory info
            auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
            snapshot.cuda_allocated_bytes = stats.allocated_bytes[0].current;
            snapshot.cuda_reserved_bytes = stats.reserved_bytes[0].current;
            snapshot.cuda_active_bytes = stats.active_bytes[0].current;
            // Calculate inactive as the difference between reserved and allocated
            snapshot.cuda_inactive_bytes = snapshot.cuda_reserved_bytes > snapshot.cuda_allocated_bytes
                                               ? snapshot.cuda_reserved_bytes - snapshot.cuda_allocated_bytes
                                               : 0;

            // System CUDA memory
            cudaMemGetInfo(&snapshot.system_free_bytes, &snapshot.system_total_bytes);

            if (enabled_) {
                snapshots_.push_back(snapshot);

                // Log if detailed mode
                if (detailed_) {
                    log_snapshot(snapshot);
                }
            }

            return snapshot;
        }

        void log_snapshot(const MemorySnapshot& s) {
            auto elapsed = std::chrono::duration<double>(s.timestamp - start_time_).count();

            LOG_INFO("[Memory] Iter {}: {} | "
                     "Allocated: {:.2f}GB | Reserved: {:.2f}GB | "
                     "Active: {:.2f}GB | Inactive: {:.2f}GB | "
                     "GPU Free: {:.2f}GB/{:.2f}GB | "
                     "Time: {:.2f}s",
                     s.iteration, s.phase,
                     bytes_to_gb(s.cuda_allocated_bytes),
                     bytes_to_gb(s.cuda_reserved_bytes),
                     bytes_to_gb(s.cuda_active_bytes),
                     bytes_to_gb(s.cuda_inactive_bytes),
                     bytes_to_gb(s.system_free_bytes),
                     bytes_to_gb(s.system_total_bytes),
                     elapsed);
        }

        void log_tensor_info(const std::string& name, const torch::Tensor& tensor) {
            if (!enabled_ || !detailed_)
                return;

            size_t bytes = tensor.numel() * tensor.element_size();

            // Convert tensor properties to strings to avoid formatter issues
            std::stringstream shape_ss;
            shape_ss << tensor.sizes();

            std::stringstream dtype_ss;
            dtype_ss << tensor.dtype();

            std::stringstream device_ss;
            device_ss << tensor.device();

            LOG_DEBUG("  Tensor '{}': shape={}, dtype={}, device={}, size={:.2f}MB",
                      name,
                      shape_ss.str(),
                      dtype_ss.str(),
                      device_ss.str(),
                      bytes_to_mb(bytes));
        }

        void log_model_memory(const SplatData& model) {
            if (!enabled_)
                return;

            size_t total_bytes = 0;

            auto log_and_sum = [&](const std::string& name, const torch::Tensor& t) {
                if (t.defined()) {
                    size_t bytes = t.numel() * t.element_size();
                    total_bytes += bytes;
                    if (detailed_) {
                        LOG_DEBUG("  Model.{}: {:.2f}MB", name, bytes_to_mb(bytes));
                    }
                }
            };

            LOG_INFO("[Memory] Model tensors:");
            log_and_sum("means", model.means());
            log_and_sum("sh0", model.sh0());
            log_and_sum("shN", model.shN());
            log_and_sum("scaling", model.scaling_raw());
            log_and_sum("rotation", model.rotation_raw());
            log_and_sum("opacity", model.opacity_raw());
            log_and_sum("densification_info", model._densification_info);

            LOG_INFO("[Memory] Total model size: {:.2f}MB for {} Gaussians",
                     bytes_to_mb(total_bytes), model.size());
        }

        void log_optimizer_memory(torch::optim::Optimizer* optimizer) {
            if (!enabled_ || !optimizer)
                return;

            size_t total_state_bytes = 0;
            int state_count = 0;

            for (const auto& [key, state] : optimizer->state()) {
                state_count++;

                // Try to cast to different optimizer states
                if (auto* adam_state = dynamic_cast<FusedAdam::AdamParamState*>(state.get())) {
                    if (adam_state->exp_avg.defined()) {
                        total_state_bytes += adam_state->exp_avg.numel() * adam_state->exp_avg.element_size();
                    }
                    if (adam_state->exp_avg_sq.defined()) {
                        total_state_bytes += adam_state->exp_avg_sq.numel() * adam_state->exp_avg_sq.element_size();
                    }
                    if (adam_state->max_exp_avg_sq.defined()) {
                        total_state_bytes += adam_state->max_exp_avg_sq.numel() * adam_state->max_exp_avg_sq.element_size();
                    }
                }
            }

            LOG_INFO("[Memory] Optimizer: {} states, ~{:.2f}MB total",
                     state_count, bytes_to_mb(total_state_bytes));
        }

        void save_report(const std::filesystem::path& output_path) {
            if (snapshots_.empty())
                return;

            auto report_path = output_path / "memory_report.csv";
            std::ofstream file(report_path);

            file << "iteration,phase,elapsed_s,allocated_gb,reserved_gb,active_gb,inactive_gb,gpu_free_gb,gpu_total_gb\n";

            for (const auto& s : snapshots_) {
                auto elapsed = std::chrono::duration<double>(s.timestamp - start_time_).count();
                file << s.iteration << ","
                     << s.phase << ","
                     << elapsed << ","
                     << bytes_to_gb(s.cuda_allocated_bytes) << ","
                     << bytes_to_gb(s.cuda_reserved_bytes) << ","
                     << bytes_to_gb(s.cuda_active_bytes) << ","
                     << bytes_to_gb(s.cuda_inactive_bytes) << ","
                     << bytes_to_gb(s.system_free_bytes) << ","
                     << bytes_to_gb(s.system_total_bytes) << "\n";
            }

            LOG_INFO("Memory report saved to: {}", report_path.string());
        }

        void check_memory_pressure() {
            if (!enabled_)
                return;

            size_t free, total;
            cudaMemGetInfo(&free, &total);

            float usage_percent = 100.0f * (1.0f - static_cast<float>(free) / total);

            if (usage_percent > 90.0f) {
                LOG_WARN("[Memory] High GPU memory usage: {:.1f}% ({:.2f}GB free of {:.2f}GB)",
                         usage_percent, bytes_to_gb(free), bytes_to_gb(total));

                // Log cache allocator stats
                auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);

                // Calculate inactive bytes as difference between reserved and allocated
                size_t inactive_bytes = stats.reserved_bytes[0].current > stats.allocated_bytes[0].current
                                            ? stats.reserved_bytes[0].current - stats.allocated_bytes[0].current
                                            : 0;

                LOG_WARN("[Memory] Cache allocator - Allocated: {:.2f}GB, Reserved: {:.2f}GB, Inactive: {:.2f}GB",
                         bytes_to_gb(stats.allocated_bytes[0].current),
                         bytes_to_gb(stats.reserved_bytes[0].current),
                         bytes_to_gb(inactive_bytes));
            }
        }

        void force_cleanup() {
            if (!enabled_)
                return;

            LOG_INFO("[Memory] Forcing cache cleanup...");
            auto before = capture(0, "before_cleanup");

            c10::cuda::CUDACachingAllocator::emptyCache();
            torch::cuda::synchronize();

            auto after = capture(0, "after_cleanup");

            size_t freed = before.cuda_reserved_bytes - after.cuda_reserved_bytes;
            LOG_INFO("[Memory] Freed {:.2f}MB from cache", bytes_to_mb(freed));
        }

    private:
        MemoryTracker() = default;

        static double bytes_to_mb(size_t bytes) {
            return static_cast<double>(bytes) / (1024.0 * 1024.0);
        }

        static double bytes_to_gb(size_t bytes) {
            return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
        }

        bool enabled_ = false;
        bool detailed_ = false;
        std::vector<MemorySnapshot> snapshots_;
        std::chrono::steady_clock::time_point start_time_;
    };

    // RAII helper for tracking memory in a scope
    class ScopedMemoryTracker {
    public:
        ScopedMemoryTracker(int iteration, const std::string& phase, bool log_on_exit = true)
            : iteration_(iteration),
              phase_(phase),
              log_on_exit_(log_on_exit) {
            start_ = MemoryTracker::get().capture(iteration, phase + "_start");
        }

        ~ScopedMemoryTracker() {
            if (log_on_exit_) {
                auto end = MemoryTracker::get().capture(iteration_, phase_ + "_end");

                // Use signed arithmetic to handle both increases and decreases
                ssize_t allocated_delta = static_cast<ssize_t>(end.cuda_allocated_bytes) -
                                          static_cast<ssize_t>(start_.cuda_allocated_bytes);

                if (std::abs(allocated_delta) > 1024 * 1024) { // > 1MB change
                    LOG_DEBUG("[Memory] {} changed allocation by {:.2f}MB",
                              phase_,
                              static_cast<double>(allocated_delta) / (1024.0 * 1024.0));
                }
            }
        }

    private:
        int iteration_;
        std::string phase_;
        bool log_on_exit_;
        MemorySnapshot start_;
    };

} // namespace gs::training