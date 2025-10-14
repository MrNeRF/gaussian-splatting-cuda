/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
#include <memory>
#include <mutex>
#include <set>

namespace gs::loader {
    struct LoadParams {
        int resize_factor;
        int max_width;
    };

    class CacheLoader {
    public:
        // Delete copy constructor and assignment operator
        CacheLoader(const CacheLoader&) = delete;
        CacheLoader& operator=(const CacheLoader&) = delete;

        // Static method to get the singleton instance
        static CacheLoader& getInstance(
            const std::filesystem::path& cache_folder,
            bool use_cpu_memory,
            bool use_fs_cache) {
            std::call_once(init_flag_, [&]() {
                instance_.reset(new CacheLoader(cache_folder, use_cpu_memory, use_fs_cache));
            });
            return *instance_;
        }

        // Static method to get existing instance (throws if not initialized)
        static CacheLoader& getInstance() {
            if (!instance_) {
                throw std::runtime_error("CacheLoader not initialized. Call getInstance with parameters first.");
            }
            return *instance_;
        }

        // Main method - to be implemented
        std::tuple<unsigned char*, int, int, int> load_cached_image(const std::filesystem::path& path, const LoadParams& params);

    private:
        std::tuple<unsigned char*, int, int, int> CacheLoader::load_cached_image_from_cpu(const std::filesystem::path& path, const LoadParams& params);
        std::tuple<unsigned char*, int, int, int> CacheLoader::load_cached_image_from_fs(const std::filesystem::path& path, const LoadParams& params);
        // Private constructor
        CacheLoader(
            const std::filesystem::path& cache_folder,
            bool use_cpu_memory,
            bool use_fs_cache) : cache_folder_(cache_folder),
                                 use_cpu_memory_(use_cpu_memory),
                                 use_fs_cache_(use_fs_cache) {
            // Initialize your cache loader here
        }

        // CPU cache params
        bool use_cpu_memory_;
        float min_cpu_free_memory_ratio_ = 0.1; // make sure at least 10% RAM is free
        std::size_t min_cpu_free_GB_ = 1;       // min GB we want to be free

        // FS cache params
        std::filesystem::path cache_folder_;
        bool use_fs_cache_;

        // Singleton instance and initialization flag
        static std::unique_ptr<CacheLoader> instance_;
        static std::once_flag init_flag_;

        std::mutex cache_mutex_;
        std::set<std::string> image_being_saved_;
    };
} // namespace gs::loader