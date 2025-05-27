#pragma once

#include "core/camera_info.hpp"
#include "core/camera.hpp"
#include "core/scene_info.hpp"
#include "core/parameters.hpp"
#include "core/camera_utils.hpp"
#include "core/read_utils.hpp"
#include <torch/torch.h>
#include <memory>
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
     * LibTorch’s stateless dataloader takes the dataset **by value**, so a
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

        const auto& cam_info = _camera_infos[index];

        // Load image on demand
        auto [img_data, width, height, channels] = read_image(
            cam_info._image_path,
            _params.resolution);

        // Create tensor from image data
        torch::Tensor image_tensor = torch::from_blob(
            img_data,
            {height, width, channels},
            {width * channels, channels, 1},
            torch::kUInt8);

        // Convert to float and normalize
        image_tensor = image_tensor.to(torch::kFloat32)
                           .permute({2, 0, 1})
                           .clone() / 255.0f;

        // Free the image data
        free_image(img_data);

        // Create Camera object
        Camera camera(
            cam_info._camera_ID,
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
    std::vector<CameraInfo>    _camera_infos;
    const gs::param::ModelParameters& _params;
};

// -----------------------------------------------------------------------------
// Factory function to create a LibTorch DataLoader
// -----------------------------------------------------------------------------
inline auto create_torch_dataloader(
    const gs::param::ModelParameters& params,
    int num_workers = 4) {

    if (!std::filesystem::exists(params.source_path)) {
        throw std::runtime_error("Data path does not exist: " +
                                 params.source_path.string());
    }

    // Read scene info
    auto scene_info = read_colmap_scene_info(params.source_path, params.resolution);

    // Create dataset
    auto dataset = std::make_shared<CameraDataset>(std::move(scene_info), params);

    // Store dataset size before moving
    const size_t dataset_size = dataset->size().value();

    // DataLoader options
    auto options = torch::data::DataLoaderOptions()
                       .batch_size(1)      // We want one camera at a time
                       .workers(num_workers)
                       .enforce_ordering(false);  // Allow out‑of‑order delivery

    // Build dataloader with the desired sampler
    auto dataloader = torch::data::make_data_loader<CameraDataset>(
        *dataset,
        torch::data::samplers::RandomSampler(dataset_size),
        options);
    return std::make_tuple(std::move(dataloader), dataset);
}
