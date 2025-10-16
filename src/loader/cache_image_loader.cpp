/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "loader/cache_image_loader.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "project/project.hpp"

#include <fstream>

// Platform-specific includes
#ifdef __linux__
#include <sys/sysinfo.h>
#elif defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>
#endif

namespace gs::system {
    /**
     * @brief Get total physical memory in bytes
     * @return Total physical RAM in bytes, or default value if unavailable
     */
    std::size_t get_total_physical_memory();

    /**
     * @brief Get available physical memory in bytes
     * @return Available physical RAM in bytes, or default value if unavailable
     */
    std::size_t get_available_physical_memory();

    /**
     * @brief Get memory usage percentage (0.0 to 1.0)
     * @return Ratio of used memory to total memory
     */
    double get_memory_usage_ratio();
} // namespace gs::system


namespace gs::system {

    std::size_t get_total_physical_memory() {
#ifdef __linux__
        struct sysinfo info;
        if (sysinfo(&info) == 0) {
            return info.totalram * info.mem_unit;
        }
        LOG_WARN("Failed to get total memory on Linux");
#elif defined(_WIN32)
        MEMORYSTATUSEX mem_info;
        mem_info.dwLength = sizeof(MEMORYSTATUSEX);
        if (GlobalMemoryStatusEx(&mem_info)) {
            return mem_info.ullTotalPhys;
        }
        LOG_WARN("Failed to get total memory on Windows");
#elif defined(__APPLE__)
        int64_t mem_size;
        size_t len = sizeof(mem_size);
        if (sysctlbyname("hw.memsize", &mem_size, &len, NULL, 0) == 0) {
            return static_cast<std::size_t>(mem_size);
        }
        LOG_WARN("Failed to get total memory on macOS");
#else
        LOG_WARN("Unsupported platform for memory detection");
#endif
        // Fallback: assume 16GB
        return 16ULL * 1024 * 1024 * 1024;
    }

    std::size_t get_available_physical_memory() {
#ifdef __linux__
        struct sysinfo info;
        if (sysinfo(&info) == 0) {
            return info.freeram * info.mem_unit;
        }
        LOG_WARN("Failed to get available memory on Linux");
#elif defined(_WIN32)
        MEMORYSTATUSEX mem_info;
        mem_info.dwLength = sizeof(MEMORYSTATUSEX);
        if (GlobalMemoryStatusEx(&mem_info)) {
            return mem_info.ullAvailPhys;
        }
        LOG_WARN("Failed to get available memory on Windows");
#elif defined(__APPLE__)
        vm_statistics64_data_t vm_stats;
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                              (host_info64_t)&vm_stats, &count) == KERN_SUCCESS) {
            return (vm_stats.free_count + vm_stats.inactive_count) * vm_page_size;
        }
        LOG_WARN("Failed to get available memory on macOS");
#else
        LOG_WARN("Unsupported platform for memory detection");
#endif
        // Fallback: assume 1GB available
        return 1ULL * 1024 * 1024 * 1024;
    }

    double get_memory_usage_ratio() {
        std::size_t total = get_total_physical_memory();
        std::size_t available = get_available_physical_memory();

        if (total == 0) {
            return 1.0; // Assume full if we can't determine
        }

        return 1.0 - (static_cast<double>(available) / static_cast<double>(total));
    }

} // namespace gs::system

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

    CacheLoader::CacheLoader(bool use_cpu_memory, bool use_fs_cache) : use_cpu_memory_(use_cpu_memory), use_fs_cache_(use_fs_cache) {
        create_new_cache_folder();
        if (min_cpu_free_memory_ratio_ < 0 || min_cpu_free_memory_ratio_ > 1) {
            LOG_WARN("min_cpu_free_memory_ratio_ is outside [0,1] interval = {}",min_cpu_free_memory_ratio_);
            min_cpu_free_memory_ratio_ = std::clamp(min_cpu_free_memory_ratio_, 0.0f, 1.0f);
        }
    }

    void CacheLoader::create_new_cache_folder() {
        if (!use_fs_cache_) {
            return;
        }

        auto cache_base = gs::management::GetLichtFeldBaseTemporaryFolder() / "cache";

        if (!std::filesystem::exists(cache_base)) {
            bool success = std::filesystem::create_directories(cache_base);
            if (!success) {
                throw std::runtime_error("failed to create cache base directory " + cache_base.string());
            }
        }

        std::string unique_cache_path = LFS_CACHE_PREFIX + gs::management::generateShortHash();
        std::filesystem::path cache_folder = cache_base / unique_cache_path;

        bool success = std::filesystem::create_directories(cache_folder);
        if (!success) {
            throw std::runtime_error("failed to create cache directory " + cache_folder.string());
        }
        cache_folder_ = cache_folder;
    }

    void CacheLoader::clean_cache_folders() {
        auto cache_base = gs::management::GetLichtFeldBaseTemporaryFolder() / "cache";
        if (!std::filesystem::exists(cache_base) || !std::filesystem::is_directory(cache_base)) {
            LOG_ERROR("Invalid base folder: {}", cache_base.string());
            return;
        }

        for (const auto& entry : std::filesystem::directory_iterator(cache_base)) {
            if (entry.is_directory()) {
                auto folder_name = entry.path().filename().string();

                if (folder_name.rfind(LFS_CACHE_PREFIX, 0) == 0) { // starts with prefix
                    if (std::filesystem::exists(entry.path() / ".lock")) {
                        LOG_DEBUG("folder: {} exists, but it is locked", entry.path().string());
                        continue;
                    }
                    std::error_code ec;
                    std::filesystem::remove_all(entry.path(), ec);
                    if (ec) {
                        LOG_ERROR("Failed to remove {}:{}", entry.path().string(), ec.message());
                        continue;
                    }
                    LOG_DEBUG("Removed folder: {}", entry.path().string());
                }
            }
        }
    }

    void CacheLoader::clear_cpu_cache() {
        std::lock_guard<std::mutex> lock(cpu_cache_mutex_);
        cpu_cache_.clear();
    }

    bool CacheLoader::has_sufficient_memory(std::size_t required_bytes) const {
        std::size_t available = gs::system::get_available_physical_memory();
        std::size_t total = gs::system::get_total_physical_memory();

        std::size_t min_free_bytes = std::max(
            static_cast<std::size_t>(total * min_cpu_free_memory_ratio_),
            static_cast<std::size_t>(min_cpu_free_GB_ * 1024ULL * 1024 * 1024));

        // Check if adding this image would leave sufficient free memory
        return (available > required_bytes + min_free_bytes);
    }

    void CacheLoader::evict_until_statisfied() {

        while (!cpu_cache_.empty()) {
            std::size_t available = gs::system::get_available_physical_memory();
            std::size_t total = gs::system::get_total_physical_memory();
            std::size_t min_free_bytes = std::max(
                static_cast<std::size_t>(total * min_cpu_free_memory_ratio_),
                static_cast<std::size_t>(min_cpu_free_GB_ * 1024ULL * 1024 * 1024));

            if (available > min_free_bytes) {
                break;
            }
            {
                std::lock_guard<std::mutex> lock(cpu_cache_mutex_);
                auto oldest = std::min_element(cpu_cache_.begin(), cpu_cache_.end(),
                                               [](const auto& a, const auto& b) {
                                                   return a.second.last_access < b.second.last_access;
                                               });
                LOG_DEBUG("Evicting cached image {} ({} bytes) from CPU cache",
                          oldest->first, oldest->second.size_bytes);
                cpu_cache_.erase(oldest);
            }
        }
    }

    void CacheLoader::evict_if_needed(std::size_t required_bytes) {
        // LRU eviction: remove least recently accessed images until we have space
        while (!cpu_cache_.empty() && !has_sufficient_memory(required_bytes)) {
            auto oldest = std::min_element(cpu_cache_.begin(), cpu_cache_.end(),
                                           [](const auto& a, const auto& b) {
                                               return a.second.last_access < b.second.last_access;
                                           });

            if (oldest != cpu_cache_.end()) {
                LOG_DEBUG("Evicting cached image {} ({} bytes) from CPU cache",
                          oldest->first, oldest->second.size_bytes);
                cpu_cache_.erase(oldest);
            } else {
                break;
            }
        }
    }

    std::size_t CacheLoader::get_cpu_cache_size() const {
        std::size_t total = 0;
        for (const auto& [key, data] : cpu_cache_) {
            total += data.size_bytes;
        }
        return total;
    }

    std::string CacheLoader::generate_cache_key(const std::filesystem::path& path, const LoadParams& params) const {
        return std::format("{}:resize_{}_maxw_{}", path.string(), params.resize_factor, params.max_width);
    }

    std::tuple<unsigned char*, int, int, int> CacheLoader::load_cached_image_from_cpu(
        const std::filesystem::path& path,
        const LoadParams& params) {

        std::string cache_key = generate_cache_key(path, params);

        // Check if image is already in CPU cache
        {
            std::lock_guard<std::mutex> lock(cpu_cache_mutex_);
            auto it = cpu_cache_.find(cache_key);
            if (it != cpu_cache_.end()) {
                // Update last access time
                it->second.last_access = std::chrono::steady_clock::now();

                // Allocate new memory and copy cached data
                auto& cached = it->second;
                // alocation should be like load_img sinse use call img_free
                unsigned char* img_data = static_cast<unsigned char*>(std::malloc(cached.data.size()));
                std::memcpy(img_data, cached.data.data(), cached.data.size());

                LOG_DEBUG("Loaded image {} from CPU cache", cache_key);
                return {img_data, cached.width, cached.height, cached.channels};
            }
        }

        // Image not in cache, need to load it
        bool is_image_being_loaded = false;
        {
            std::lock_guard<std::mutex> lock(cpu_cache_mutex_);
            if (image_being_loaded_cpu_.find(cache_key) != image_being_loaded_cpu_.end()) {
                is_image_being_loaded = true;
            }
            if (!is_image_being_loaded) {
                image_being_loaded_cpu_.insert(cache_key);
            }
        }

        // If another thread is loading, fall back to direct load

        if (is_image_being_loaded) {
            LOG_DEBUG("Image {} is being loaded by another thread, loading directly", cache_key);
            if (use_fs_cache_) {
                return load_cached_image_from_fs(path, params);
            }
            return load_image(path, params.resize_factor, params.max_width);
        }

        // Load the image

        auto [img_data, width, height, channels] = use_fs_cache_ ? load_cached_image_from_fs(path, params) : load_image(path, params.resize_factor, params.max_width);

        if (img_data == nullptr) {
            std::lock_guard<std::mutex> lock(cpu_cache_mutex_);
            image_being_loaded_cpu_.erase(cache_key);
            throw std::runtime_error("Failed to load image: " + path.string());
        }

        // Calculate image size
        std::size_t img_size = static_cast<std::size_t>(width) * height * channels;

        // Try to cache the image in CPU memory
        {
            std::lock_guard<std::mutex> lock(cpu_cache_mutex_);

            // Check if we have sufficient memory
            if (has_sufficient_memory(img_size)) {
                // Evict old entries if needed
                evict_if_needed(img_size);

                // Store in cache
                CachedImageData cached_data;
                cached_data.data.resize(img_size);
                std::memcpy(cached_data.data.data(), img_data, img_size);
                cached_data.width = width;
                cached_data.height = height;
                cached_data.channels = channels;
                cached_data.size_bytes = img_size;
                cached_data.last_access = std::chrono::steady_clock::now();

                cpu_cache_[cache_key] = std::move(cached_data);

                LOG_DEBUG("Cached image {} in CPU memory ({} bytes, total cache: {} bytes)",
                          cache_key, img_size, get_cpu_cache_size());
            } else {
                LOG_DEBUG("Insufficient memory to cache image {} ({} bytes required)",
                          cache_key, img_size);
            }

            image_being_loaded_cpu_.erase(cache_key);
        }

        evict_until_statisfied();

        return {img_data, width, height, channels};
    }

    std::tuple<unsigned char*, int, int, int> CacheLoader::load_cached_image_from_fs(const std::filesystem::path& path, const LoadParams& params) {
        if (cache_folder_.empty()) {
            return load_image(path, params.resize_factor, params.max_width);
        }

        std::string unique_name = std::format("rf_{}_mw_{}_", params.resize_factor, params.max_width) + path.filename().string();

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
                LOG_INFO("successfully saved image {} in cache folder", cache_img_path.string());
                success = create_done_file(cache_img_path);

                if (!success) {
                    throw std::runtime_error("failed saving .done file " + path.filename().string() + " in cache folder " + cache_folder_.string());
                }
                image_being_saved_.erase(path.string());
            }
        }

        return result;
    }

    std::tuple<unsigned char*, int, int, int> CacheLoader::load_cached_image(const std::filesystem::path& path, const LoadParams& params) {

        if (use_cpu_memory_) {
            print_cache_status();
            return load_cached_image_from_cpu(path, params);
        }
        if (use_fs_cache_) {
            return load_cached_image_from_fs(path, params);
        }

        return load_image(path, params.resize_factor, params.max_width);
    }

    void CacheLoader::print_cache_status() const {
        if (!print_cache_status_) {
            return;
        }

        std::lock_guard<std::mutex> lock(counter_mutex_);
        load_counter_++;
        size_t giga_to_bytes = 1024ULL * 1024 * 1024;
        if (load_counter_ > print_status_freq_num_) {
            load_counter_ = 0;
            double total_memory_gb = (double)gs::system::get_total_physical_memory() / giga_to_bytes;
            double memory_ratio = gs::system::get_memory_usage_ratio();
            double cache_ratio = (double)get_cpu_cache_size() / (double)gs::system::get_total_physical_memory();

            LOG_INFO("CacheInfo: Num images in cache {}", cpu_cache_.size());
            LOG_INFO("CacheInfo: total memory {:.2f}GB", total_memory_gb);
            LOG_INFO("CacheInfo: used memory {:.2f}%", 100 * memory_ratio);
            LOG_INFO("CacheInfo: cache memory occupancy {:.2f}%", 100 * cache_ratio);
        }
    }

} // namespace gs::loader