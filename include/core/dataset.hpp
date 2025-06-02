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
    CameraDataset(std::vector<std::shared_ptr<Camera>> cameras,
                  const gs::param::DatasetConfig& params)
        : _cameras(std::move(cameras)),
          _datasetConfig(params) {

        std::cout << "CameraDataset initialized with " << _cameras.size()
                  << " cameras" << std::endl;
    }

    // Default copy constructor works with shared_ptr
    CameraDataset(const CameraDataset&) = default;
    CameraDataset(CameraDataset&&) noexcept = default;
    CameraDataset& operator=(CameraDataset&&) noexcept = default;
    CameraDataset& operator=(const CameraDataset&) = default;

    CameraExample get(size_t index) override {
        if (index >= _cameras.size()) {
            throw std::out_of_range("Camera index out of range");
        }

        auto& cam = _cameras[index];

        // Load image on demand
        torch::Tensor image = cam->load_and_get_image(_datasetConfig.resolution);

        // Return camera pointer and image
        return {{cam.get(), std::move(image)}, torch::empty({})};
    }

    torch::optional<size_t> size() const override {
        return _cameras.size();
    }

    const std::vector<std::shared_ptr<Camera>>& get_cameras() const {
        return _cameras;
    }

private:
    std::vector<std::shared_ptr<Camera>> _cameras;
    const gs::param::DatasetConfig& _datasetConfig;
};

inline std::tuple<std::shared_ptr<CameraDataset>, float> create_dataset_from_colmap(
    const gs::param::DatasetConfig& datasetConfig) {

    if (!std::filesystem::exists(datasetConfig.data_path)) {
        throw std::runtime_error("Data path does not exist: " +
                                 datasetConfig.data_path.string());
    }

    auto [camera_infos, nerf_norm] = read_colmap_cameras_and_images(datasetConfig.data_path);

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

    auto dataset = std::make_shared<CameraDataset>(std::move(cameras), datasetConfig);

    return {dataset, nerf_norm};
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