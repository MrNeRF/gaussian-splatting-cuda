#pragma once

#include "core/camera.hpp"
#include "core/parameters.hpp"
#include "loader/loader.hpp"
#include <expected>
#include <format>
#include <memory>
#include <torch/torch.h>
#include <vector>

// Camera with loaded image
namespace gs {
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

            torch::Tensor image = cam->load_and_get_image(_datasetConfig.resolution);
            return {{cam.get(), std::move(image)}, torch::empty({})};
        }

        torch::optional<size_t> size() const override {
            return _indices.size();
        }

        const std::vector<std::shared_ptr<Camera>>& get_cameras() const {
            return _cameras;
        }

        Split get_split() const { return _split; }

        size_t get_num_bytes() const {
            if (_cameras.empty()) {
                return 0;
            }
            size_t total_bytes = 0;
            for (const auto& cam : _cameras) {
                total_bytes += cam->get_num_bytes_from_file();
            }
            // Adjust for resolution factor if specified
            if (_datasetConfig.resolution > 0) {
                total_bytes /= _datasetConfig.resolution * _datasetConfig.resolution;
            }
            return total_bytes;
        }

    private:
        std::vector<std::shared_ptr<Camera>> _cameras;
        const gs::param::DatasetConfig _datasetConfig;
        Split _split;
        std::vector<size_t> _indices;
    };

    // Infinite random sampler for continuous data flow
    class InfiniteRandomSampler : public torch::data::samplers::RandomSampler {
    public:
        using super = torch::data::samplers::RandomSampler;

        explicit InfiniteRandomSampler(size_t dataset_size)
            : super(dataset_size) {}

        std::optional<std::vector<size_t>> next(size_t batch_size) override {
            auto indices = super::next(batch_size);
            if (!indices) {
                super::reset();
                indices = super::next(batch_size);
            }
            return indices;
        }

    private:
        size_t dataset_size_;
    };

    inline std::expected<std::tuple<std::shared_ptr<CameraDataset>, torch::Tensor>, std::string>
    create_dataset_from_colmap(const gs::param::DatasetConfig& datasetConfig) {

        try {
            if (!std::filesystem::exists(datasetConfig.data_path)) {
                return std::unexpected(std::format("Data path does not exist: {}",
                                                   datasetConfig.data_path.string()));
            }

            // Create loader
            auto loader = gs::loader::Loader::create();

            // Set up load options
            gs::loader::LoadOptions options{
                .resolution = datasetConfig.resolution,
                .images_folder = datasetConfig.images,
                .validate_only = false};

            // Load the data
            auto result = loader->load(datasetConfig.data_path, options);
            if (!result) {
                return std::unexpected(std::format("Failed to load COLMAP dataset: {}", result.error()));
            }

            // Handle the result
            return std::visit([&datasetConfig, &result](auto&& data) -> std::expected<std::tuple<std::shared_ptr<CameraDataset>, torch::Tensor>, std::string> {
                using T = std::decay_t<decltype(data)>;

                if constexpr (std::is_same_v<T, std::shared_ptr<gs::SplatData>>) {
                    return std::unexpected("Expected COLMAP dataset but got PLY file");
                } else if constexpr (std::is_same_v<T, gs::loader::LoadedScene>) {
                    if (!data.cameras) {
                        return std::unexpected("Loaded scene has no cameras");
                    }
                    // Return the cameras that were already loaded
                    return std::make_tuple(data.cameras, result->scene_center);
                } else {
                    return std::unexpected("Unknown data type returned from loader");
                }
            },
                              result->data);

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to create dataset from COLMAP: {}", e.what()));
        }
    }

    inline std::expected<std::tuple<std::shared_ptr<CameraDataset>, torch::Tensor>, std::string>
    create_dataset_from_transforms(const gs::param::DatasetConfig& datasetConfig) {

        try {
            if (!std::filesystem::exists(datasetConfig.data_path)) {
                return std::unexpected(std::format("Data path does not exist: {}",
                                                   datasetConfig.data_path.string()));
            }

            // Create loader
            auto loader = gs::loader::Loader::create();

            // Set up load options
            gs::loader::LoadOptions options{
                .resolution = datasetConfig.resolution,
                .images_folder = datasetConfig.images,
                .validate_only = false};

            // Load the data
            auto result = loader->load(datasetConfig.data_path, options);
            if (!result) {
                return std::unexpected(std::format("Failed to load transforms dataset: {}", result.error()));
            }

            // Handle the result
            return std::visit([&datasetConfig, &result](auto&& data) -> std::expected<std::tuple<std::shared_ptr<CameraDataset>, torch::Tensor>, std::string> {
                using T = std::decay_t<decltype(data)>;

                if constexpr (std::is_same_v<T, std::shared_ptr<gs::SplatData>>) {
                    return std::unexpected("Expected transforms.json dataset but got PLY file");
                } else if constexpr (std::is_same_v<T, gs::loader::LoadedScene>) {
                    if (!data.cameras) {
                        return std::unexpected("Loaded scene has no cameras");
                    }
                    // Return the cameras that were already loaded
                    return std::make_tuple(data.cameras, result->scene_center);
                } else {
                    return std::unexpected("Unknown data type returned from loader");
                }
            },
                              result->data);

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to create dataset from transforms: {}", e.what()));
        }
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

    inline auto create_infinite_dataloader_from_dataset(
        std::shared_ptr<CameraDataset> dataset,
        int num_workers = 4) {

        const size_t dataset_size = dataset->size().value();

        return torch::data::make_data_loader(
            *dataset,
            InfiniteRandomSampler(dataset_size),
            torch::data::DataLoaderOptions()
                .batch_size(1)
                .workers(num_workers)
                .enforce_ordering(false));
    }
} // namespace gs