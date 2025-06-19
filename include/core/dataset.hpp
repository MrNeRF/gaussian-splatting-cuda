#pragma once

#include "core/camera.hpp"
#include "core/colmap_reader.hpp"
#include "core/frequency_scheduler.hpp"
#include "core/parameters.hpp"
#include <atomic>
#include <memory>
#include <torch/torch.h>
#include <vector>

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
                  Split split = Split::ALL,
                  std::shared_ptr<gs::FrequencyScheduler> freq_scheduler = nullptr)
        : _cameras(std::move(cameras)),
          _datasetConfig(params), // Make a copy
          _split(split),
          _freq_scheduler(freq_scheduler),
          _current_iteration(std::make_shared<std::atomic<int>>(0)) {

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

    // Default copy/move constructors and assignment operators work now
    CameraDataset(const CameraDataset&) = default;
    CameraDataset(CameraDataset&&) noexcept = default;
    CameraDataset& operator=(const CameraDataset&) = default;
    CameraDataset& operator=(CameraDataset&&) noexcept = default;

    CameraExample get(size_t index) override {
        if (index >= _indices.size()) {
            throw std::out_of_range("Dataset index out of range");
        }

        size_t camera_idx = _indices[index];
        auto& cam = _cameras[camera_idx];

        torch::Tensor image;

        // Check if we should use frequency scheduling (only for training)
        if (_freq_scheduler && _freq_scheduler->is_enabled() && _split == Split::TRAIN) {
            float factor = _freq_scheduler->get_factor_for_iteration(_current_iteration->load());
            image = cam->load_and_get_image_with_factor(factor);
        } else {
            // Normal loading (full resolution or specified resolution)
            image = cam->load_and_get_image(_datasetConfig.resolution);
        }

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

    // Update the current iteration (called by trainer)
    void update_iteration(int iter) {
        _current_iteration->store(iter);
    }

private:
    std::vector<std::shared_ptr<Camera>> _cameras;
    gs::param::DatasetConfig _datasetConfig; // Store a copy instead of const reference
    Split _split;
    std::vector<size_t> _indices;
    std::shared_ptr<gs::FrequencyScheduler> _freq_scheduler;
    std::shared_ptr<std::atomic<int>> _current_iteration; // Shared pointer for copyability
};

inline std::tuple<std::shared_ptr<CameraDataset>, float, std::shared_ptr<gs::FrequencyScheduler>>
create_dataset_from_colmap(const gs::param::DatasetConfig& datasetConfig,
                           const gs::param::OptimizationParameters& optParams) {

    if (!std::filesystem::exists(datasetConfig.data_path)) {
        throw std::runtime_error("Data path does not exist: " +
                                 datasetConfig.data_path.string());
    }

    // Read COLMAP data with specified images folder
    auto [camera_infos, scene_scale] = read_colmap_cameras_and_images(
        datasetConfig.data_path, datasetConfig.images);

    std::vector<std::shared_ptr<Camera>> cameras;
    cameras.reserve(camera_infos.size());

    std::vector<std::filesystem::path> image_paths;
    image_paths.reserve(camera_infos.size());

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

        cameras.push_back(std::move(cam));
        image_paths.push_back(info._image_path);
    }

    // Initialize frequency scheduler if enabled
    std::shared_ptr<gs::FrequencyScheduler> freq_scheduler = nullptr;
    if (optParams.use_frequency_schedule) {
        freq_scheduler = std::make_shared<gs::FrequencyScheduler>();
        std::cout << "Initializing frequency-based resolution scheduler..." << std::endl;
        freq_scheduler->initialize(image_paths, optParams.iterations);
    }

    // Create dataset with ALL images
    auto dataset = std::make_shared<CameraDataset>(
        std::move(cameras), datasetConfig, CameraDataset::Split::ALL, freq_scheduler);

    return {dataset, scene_scale, freq_scheduler};
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