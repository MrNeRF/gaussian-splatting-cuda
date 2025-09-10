/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/parameters.hpp"
#include "loader/loader.hpp"
#include <expected>
#include <format>
#include <memory>
#include <torch/torch.h>
#include <vector>

namespace gs::training {
    struct CameraWithImage {
        Camera* camera;
        torch::Tensor image;
        torch::Tensor attentionMask;
    };

    class CameraDataset {
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

        // Simple get method - returns our struct directly
        CameraWithImage get(size_t index) {
            if (index >= _indices.size()) {
                throw std::out_of_range("Dataset index out of range");
            }

            size_t camera_idx = _indices[index];
            auto& cam = _cameras[camera_idx];

            torch::Tensor image = cam->load_and_get_image(_datasetConfig.resize_factor);
            torch::Tensor attention_weights = cam->load_and_get_attention_weights(_datasetConfig.resize_factor);
            return {cam.get(), std::move(image), std::move(attention_weights)};
        }

        size_t size() const { return _indices.size(); }

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
            if (_datasetConfig.resize_factor > 0) {
                total_bytes /= _datasetConfig.resize_factor * _datasetConfig.resize_factor;
            }
            return total_bytes;
        }

        std::optional<Camera*> get_camera_by_filename(const std::string& filename) const {
            for (const auto& cam : _cameras) {
                if (cam->image_name() == filename) {
                    return cam.get();
                }
            }
            return std::nullopt;
        }

        void set_resize_factor(int resize_factor) { _datasetConfig.resize_factor = resize_factor; }
        int get_resize_factor() const { return _datasetConfig.resize_factor; }

    private:
        std::vector<std::shared_ptr<Camera>> _cameras;
        gs::param::DatasetConfig _datasetConfig;
        Split _split;
        std::vector<size_t> _indices;
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
                .resize_factor = datasetConfig.resize_factor,
                .images_folder = datasetConfig.images,
                .validate_only = false};

            // Load the data
            auto result = loader->load(datasetConfig.data_path, options);
            if (!result) {
                return std::unexpected(std::format("Failed to load COLMAP dataset: {}", result.error()));
            }

            // Handle the result
            return std::visit(
                [&result](
                    auto&& data) -> std::expected<std::tuple<std::shared_ptr<CameraDataset>, torch::Tensor>, std::string> {
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
                .resize_factor = datasetConfig.resize_factor,
                .images_folder = datasetConfig.images,
                .validate_only = false};

            // Load the data
            auto result = loader->load(datasetConfig.data_path, options);
            if (!result) {
                return std::unexpected(std::format("Failed to load transforms dataset: {}", result.error()));
            }

            // Handle the result
            return std::visit(
                [&datasetConfig, &result](
                    auto&& data) -> std::expected<std::tuple<std::shared_ptr<CameraDataset>, torch::Tensor>, std::string> {
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

    // Forward declarations for our dataloaders
    class InfiniteDataLoaderWrapper;
    class EvalDataLoader;

    // Factory functions
    std::unique_ptr<InfiniteDataLoaderWrapper>
    create_efficient_infinite_dataloader(
        std::shared_ptr<CameraDataset> dataset,
        int num_workers = 4);

    std::unique_ptr<EvalDataLoader>
    create_eval_dataloader(
        std::shared_ptr<CameraDataset> dataset);

} // namespace gs::training