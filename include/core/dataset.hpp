#pragma once

#include "core/camera.hpp"
#include "core/camera_info.hpp"
#include "core/camera_utils.hpp"
#include "core/parameters.hpp"
#include "core/read_utils.hpp"
#include "core/scene_info.hpp"
#include <memory>
#include <torch/torch.h>
#include <vector>

// Example type that wraps Camera
using CameraExample = torch::data::Example<Camera, torch::Tensor>;

// Custom Dataset class following LibTorch patterns
class CameraDataset : public torch::data::Dataset<CameraDataset, CameraExample> {
public:
    /**
     * Primary constructor – takes ownership of the SceneInfo that was just
     * read from COLMAP and stores a const‑reference to the global parameters
     */
    CameraDataset(std::unique_ptr<SceneInfo> scene_info,
                  const gs::param::ModelParameters& params)
        : _scene_info(std::move(scene_info)),
          _params(params) {

        // Extract camera infos from scene
        _camera_infos = std::move(_scene_info->_cameras);

        std::cout << "CameraDataset initialized with " << _camera_infos.size()
                  << " cameras" << std::endl;
    }

    /**
     * Deep‑copy constructor.
     *
     * LibTorch's stateless dataloader takes the dataset **by value**, so a
     * copy is required.  The SceneInfo held through a unique_ptr is deep‑copied
     * here to keep ownership semantics intact.  The parameters object is
     * *not* copied (it is an immutable external object) – the reference is
     * simply rebound.
     */
    CameraDataset(const CameraDataset& other)
        : _scene_info(std::make_unique<SceneInfo>(*other._scene_info)),
          _camera_infos(other._camera_infos),
          _params(other._params) {}

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
        cam_info.load_image_data(_params.resolution);

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

    // Get scene info
    const SceneInfo& get_scene_info() const {
        return *_scene_info;
    }

private:
    std::unique_ptr<SceneInfo> _scene_info;
    std::vector<CameraInfo> _camera_infos;
    const gs::param::ModelParameters& _params;
};

inline std::shared_ptr<CameraDataset> create_camera_dataset(
    const gs::param::ModelParameters& params) {

    if (!std::filesystem::exists(params.source_path)) {
        throw std::runtime_error("Data path does not exist: " +
                                 params.source_path.string());
    }

    // Read scene info (now without loading image data)
    auto scene_info = read_colmap_scene_info(params.source_path);

    // Create and return dataset
    return std::make_shared<CameraDataset>(std::move(scene_info), params);
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