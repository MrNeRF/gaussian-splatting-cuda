#pragma once

#include "core/camera.hpp"
#include "core/colmap_reader.hpp"
#include "core/parameters.hpp"
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
        torch::Tensor image = cam->load_and_get_image(_datasetConfig.resolution);

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