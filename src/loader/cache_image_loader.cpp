/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "cache_image_loader.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"

#include <fstream>

namespace gs::loader {
    std::unique_ptr<CacheLoader> CacheLoader::instance_ = nullptr;
    std::once_flag CacheLoader::init_flag_;

    static bool create_done_file(const std::filesystem::path& img_path) {
        try {
            auto done_path = img_path;
            done_path += ".done";

            // Create or overwrite the .done file
            std::ofstream ofs(done_path, std::ios::trunc);
            if (!ofs) {
                return false; // Failed to open for writing
            }

            return true;
        } catch (const std::exception& e) {
            LOG_ERROR("Exception in create_done_file for {} img_path : {} ", img_path.string(), e.what());
            return false;
        } catch (...) {
            LOG_ERROR("Unknown exception in create_done_file for {}", img_path.string());
            return false;
        }
    }

    static bool does_cache_image_exists(const std::filesystem::path& img_path) {
        // Assume cached image has same base name but with ".done" marker
        auto done_path = img_path;
        done_path += ".done";

        // Both the main file and its ".done" marker must exist
        return std::filesystem::exists(img_path) &&
               std::filesystem::exists(done_path);
    }

    std::tuple<unsigned char*, int, int, int> CacheLoader::load_cached_image_from_cpu(const std::filesystem::path& path, const LoadParams& params) {
    }

    std::tuple<unsigned char*, int, int, int> CacheLoader::load_cached_image_from_fs(const std::filesystem::path& path, const LoadParams& params) {
        if (cache_folder_.empty()) {
            return load_image(path, params.resize_factor, params.max_width);
        }

        std::string unique_name = std::format("resize_factor_{}_", params.resize_factor) + path.filename().string();

        auto cache_img_path = cache_folder_ / unique_name;

        std::tuple<unsigned char*, int, int, int> result;
        if (does_cache_image_exists(cache_img_path)) {
            // Load image synchronously
            result = load_image(cache_img_path);
        } else {

            result = load_image(path, params.resize_factor, params.max_width);

            // we want only one thread to save the data - if we enter this scope either we save the image or it exists
            bool is_image_being_saved = false;
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                if (image_being_saved_.find(path.string()) != image_being_saved_.end()) {
                    is_image_being_saved = true;
                }
                if (!is_image_being_saved) {
                    image_being_saved_.insert(path.string());
                }
            }

            if (!is_image_being_saved) { // only one thread should enter this skope
                bool success = save_img_data(cache_img_path, result);

                if (!success) {
                    throw std::runtime_error("failed saving image" + path.filename().string() + " in cache folder " + cache_folder_.string());
                }
                LOG_INFO("successfully saved image {} in cache", cache_img_path.string());
                success = create_done_file(cache_img_path);

                if (!success) {
                    throw std::runtime_error("failed saving .done file " + path.filename().string() + " in cache folder " + cache_folder_.string());
                }
                image_being_saved_.erase(path.string());
            }
        }
    }

    std::tuple<unsigned char*, int, int, int> CacheLoader::load_cached_image(const std::filesystem::path& path, const LoadParams& params) {
        bool is_image_in_cpu = false;
        if (use_cpu_memory_) {
        }
        if (use_fs_cache_ && !is_image_in_cpu) {

            load_cached_image_from_fs(path, params);
        }

        return load_image(path, params.max_width, params.resize_factor);
    }

} // namespace gs::loader