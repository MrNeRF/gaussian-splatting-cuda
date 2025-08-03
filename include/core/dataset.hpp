#pragma once

#include "core/camera.hpp"
#include "core/colmap_reader.hpp"
#include "core/parameters.hpp"
#include <memory>
#include <torch/torch.h>
#include <vector>
#include <cuda_runtime.h>
#include <fstream>

constexpr double   kCapFraction = 0.80;

struct MemInfo { uint64_t free_b, total_b; };

inline MemInfo gpu_mem_info() {
    size_t free_b = 0, total_b = 0;
    auto e = cudaMemGetInfo(&free_b, &total_b);
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    return MemInfo{ static_cast<uint64_t>(free_b), static_cast<uint64_t>(total_b) };
}

inline MemInfo cpu_mem_info() {
    std::ifstream f("/proc/meminfo");
    if (!f) throw std::runtime_error("Cannot open /proc/meminfo");
    uint64_t mem_total_kb = 0, mem_avail_kb = 0; std::string k, unit; uint64_t v;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        if (iss >> k >> v >> unit) {
            if (k == "MemTotal:")      mem_total_kb = v;
            else if (k == "MemAvailable:") mem_avail_kb = v;
        }
    }
    if (!mem_total_kb || !mem_avail_kb)
        throw std::runtime_error("Failed to parse MemTotal/MemAvailable");
    return MemInfo{ mem_avail_kb * 1024ULL, mem_total_kb * 1024ULL };
}

// Safe budget left to use if we cap at cap_fraction of total.
// device_used_now = total - free.
// Budget_now = max(0, cap_fraction*total - device_used_now).
inline uint64_t budget_left_now(const MemInfo& m, double cap_fraction) {
    const long double cap = cap_fraction * static_cast<long double>(m.total_b);
    const long double used = static_cast<long double>(m.total_b - m.free_b);
    long double left = cap - used;
    return left > 0 ? static_cast<uint64_t>(left) : 0ULL;
}

inline uint64_t img_bytes_float_rgb(uint32_t w, uint32_t h) {
    return static_cast<uint64_t>(w) * h * 3ULL * sizeof(float);
}

// Camera with loaded image
struct CameraWithImage {
    Camera* camera;
    torch::Tensor image;
};

using CameraExample = torch::data::Example<CameraWithImage, torch::Tensor>;

class CameraDataset : public torch::data::Dataset<CameraDataset, CameraExample> {
public:
    enum class Split {
        TRAIN,
        VAL,
        ALL
    };

    CameraDataset(std::vector<std::shared_ptr<Camera>> cameras,
                  const gs::param::DatasetConfig& params,
                  Split split = Split::ALL)
        : _cameras(std::move(cameras)),
          _datasetConfig(params),
          _split(split) {

        // Create indices based on split
        _indices.clear();
        for (size_t i = 0; i < _cameras.size(); ++i) {
            const bool is_test = (i % params.test_every) == 0;

            if (_split == Split::ALL ||
                (_split == Split::TRAIN && !is_test) ||
                (_split == Split::VAL && is_test)) {
                _indices.push_back(i);
            }
        }

        std::cout << "Dataset created with " << _indices.size()
                  << " images (split: " << static_cast<int>(_split) << ")" << std::endl;
    }
    // Default copy constructor works with shared_ptr
    CameraDataset(const CameraDataset&) = default;
    CameraDataset(CameraDataset&&) noexcept = default;
    CameraDataset& operator=(CameraDataset&&) noexcept = default;
    CameraDataset& operator=(const CameraDataset&) = default;

    CameraExample get(size_t index) override {
        if (index >= _indices.size()) {
            throw std::out_of_range("Dataset index out of range");
        }

        size_t camera_idx = _indices[index];
        auto& cam = _cameras[camera_idx];

        // Just load image - no prefetching since indices are random
        torch::Tensor image = cam->load_image_cache(_datasetConfig.resolution);

        // Return camera pointer and image
        return {{cam.get(), std::move(image)}, torch::empty({})};
    }

    torch::optional<size_t> size() const override {
        return _indices.size();
    }

    const std::vector<std::shared_ptr<Camera>>& get_cameras() const {
        return _cameras;
    }

    Split get_split() const { return _split; }

private:
    std::vector<std::shared_ptr<Camera>> _cameras;
    const gs::param::DatasetConfig& _datasetConfig;
    Split _split;
    std::vector<size_t> _indices;
};

inline std::tuple<std::shared_ptr<CameraDataset>, torch::Tensor> create_dataset_from_colmap(
    const gs::param::DatasetConfig& datasetConfig) {

    if (!std::filesystem::exists(datasetConfig.data_path)) {
        throw std::runtime_error("Data path does not exist: " +
                                 datasetConfig.data_path.string());
    }

    // Read COLMAP data with specified images folder
    auto [camera_infos, scene_center] = read_colmap_cameras_and_images(
        datasetConfig.data_path, datasetConfig.images);

    std::vector<std::shared_ptr<Camera>> cameras;
    cameras.reserve(camera_infos.size());

    for (size_t i = 0; i < camera_infos.size(); ++i) {
        const auto& info = camera_infos[i];

        auto cam = std::make_shared<Camera>(
            info._R,
            info._T,
            info._fov_x,
            info._fov_y,
            info._image_name,
            info._image_path,
            info._width,
            info._height,
            static_cast<int>(i));


        const uint64_t bytes = img_bytes_float_rgb(info._width, info._height);

        // Re check mem after each iteration
        MemInfo g = gpu_mem_info();
        MemInfo c = cpu_mem_info();

        uint64_t gpu_left_now = budget_left_now(g, kCapFraction);
        uint64_t cpu_left_now = budget_left_now(c, kCapFraction);
        
        // Decide where to cache the image based on available memory
        if (bytes <= gpu_left_now) {
            // Use GPU cache if available
            cam->cache_on_gpu = true;
            cam->cache_on_cpu = false;
        } else if (bytes <= cpu_left_now) {
            // Use CPU cache if GPU is not available or full
            cam->cache_on_gpu = false;
            cam->cache_on_cpu = true;
        } else {
            // stream from disk
            cam->cache_on_gpu = false;
            cam->cache_on_cpu = false;
        }

        cameras.push_back(std::move(cam));
    }

    // Create dataset with ALL images
    auto dataset = std::make_shared<CameraDataset>(
        std::move(cameras), datasetConfig, CameraDataset::Split::ALL);

    return {dataset, scene_center};
}

inline auto create_dataloader_from_dataset(
    std::shared_ptr<CameraDataset> dataset,
    int num_workers = 4) {

    const size_t dataset_size = dataset->size().value();

    return torch::data::make_data_loader(
        *dataset,
        torch::data::samplers::RandomSampler(dataset_size),
        torch::data::DataLoaderOptions()
            .batch_size(1)
            .workers(num_workers)
            .enforce_ordering(false));
}