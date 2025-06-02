#pragma once

#include "core/camera.hpp"
#include "core/camera_info.hpp"
#include "core/camera_utils.hpp"
#include "core/parameters.hpp"
#include "core/read_utils.hpp"
#include <memory>
#include <torch/torch.h>
#include <vector>

// Example type that wraps Camera
using CameraExample = torch::data::Example<Camera, torch::Tensor>;

// Custom Dataset class following LibTorch patterns
class CameraDataset : public torch::data::Dataset<CameraDataset, CameraExample> {
public:
    /**
     * Primary constructor – stores camera infos directly
     */
    CameraDataset(std::vector<CameraInfo> camera_infos,
                  const gs::param::DatasetConfig& params)
        : _camera_infos(std::move(camera_infos)),
          _datasetConfig(params) {

        std::cout << "CameraDataset initialized with " << _camera_infos.size()
                  << " cameras" << std::endl;
    }

    /**
     * Deep‑copy constructor.
     *
     * LibTorch's stateless dataloader takes the dataset **by value**, so a
     * copy is required.
     */
    CameraDataset(const CameraDataset& other)
        : _camera_infos(other._camera_infos),
          _datasetConfig(other._datasetConfig) {}

    // Move operations – default implementation is fine
    CameraDataset(CameraDataset&&) noexcept = default;
    CameraDataset& operator=(CameraDataset&&) noexcept = default;

    // Copy assignment is unnecessary – disable it explicitly to avoid misuse
    CameraDataset& operator=(const CameraDataset&) = delete;

    // Required: Get a single example
    CameraExample get(size_t index) override {
        if (index >= _camera_infos.size()) {
            throw std::out_of_range("Camera index out of range");
        }

        auto& cam_info = _camera_infos[index];

        // Load image on demand using the helper method
        cam_info.load_image_data(_datasetConfig.resolution);

        // Create tensor from the loaded image data
        torch::Tensor image_tensor = torch::from_blob(
            cam_info._img_data,
            {cam_info._img_h, cam_info._img_w, cam_info._channels},
            {cam_info._img_w * cam_info._channels, cam_info._channels, 1},
            torch::kUInt8);

        // Convert to float and normalize
        image_tensor = image_tensor.to(torch::kFloat32)
                           .permute({2, 0, 1})
                           .clone() / // Clone to own the memory before freeing
                       255.0f;

        // Free the image data immediately after cloning to tensor
        // This helps keep memory usage low
        cam_info.free_image_data();

        // Create Camera object
        Camera camera(
            cam_info._R,
            cam_info._T,
            cam_info._fov_x,
            cam_info._fov_y,
            std::move(image_tensor),
            cam_info._image_name,
            static_cast<int>(index));

        // Return as Example with dummy target (we don't use targets in this case)
        return {std::move(camera), torch::empty({})};
    }

    // Required: Return the size of the dataset
    torch::optional<size_t> size() const override {
        return _camera_infos.size();
    }

    // Get camera infos if needed
    const std::vector<CameraInfo>& get_camera_infos() const {
        return _camera_infos;
    }

private:
    std::vector<CameraInfo> _camera_infos;
    const gs::param::DatasetConfig& _datasetConfig;
};

// Updated factory function to work without SceneInfo
inline std::tuple<std::shared_ptr<CameraDataset>, float> create_dataset_from_colmap(
    const gs::param::DatasetConfig& datasetConfig) {

    if (!std::filesystem::exists(datasetConfig.data_path)) {
        throw std::runtime_error("Data path does not exist: " +
                                 datasetConfig.data_path.string());
    }

    // Read cameras and nerf norm
    auto [camera_infos, nerf_norm] = read_colmap_cameras_and_images(datasetConfig.data_path);

    // Create dataset
    auto dataset = std::make_shared<CameraDataset>(std::move(camera_infos), datasetConfig);

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