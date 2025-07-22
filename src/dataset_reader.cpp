#include "core/dataset_reader.hpp"

#include "core/dataset.hpp"
#include <filesystem>
#include <nlohmann/json.hpp>
#include <print>
#include <torch/torch.h>

#include "core/colmap_reader.hpp"
#include "core/transforms_reader.hpp"

std::expected<std::tuple<std::shared_ptr<CameraDataset>, torch::Tensor>, std::string>
ColmapReader::create_dataset() const {
    return create_dataset_from_colmap(m_datasetConfig);
}

PointCloud ColmapReader::createPointCloud() const {
    return read_colmap_point_cloud(m_datasetConfig.data_path); // Return default-constructed PointCloud for now
}

bool does_sparse_file_path_exists(const std::filesystem::path& base, const std::string& filename) {
    std::filesystem::path candidate0 = base / "sparse" / "0" / filename;
    if (std::filesystem::exists(candidate0))
        return true;

    std::print("could not find candidate {}", candidate0.string());

    std::filesystem::path candidate = base / "sparse" / filename;
    if (std::filesystem::exists(candidate))
        return true;
    std::println("could not find candidate {}", candidate.string());

    return false;
}

bool ColmapReader::isValid() const {
    if (!std::filesystem::exists(m_datasetConfig.data_path)) {
        // std::println("data path does not exist {}", m_datasetConfig.data_path);
        return false;
    }

    if (!does_sparse_file_path_exists(m_datasetConfig.data_path, "points3D.bin")) {
        return false;
    }

    if (!does_sparse_file_path_exists(m_datasetConfig.data_path, "cameras.bin")) {
        return false;
    }

    if (!does_sparse_file_path_exists(m_datasetConfig.data_path, "images.bin")) {
        return false;
    }

    return true;
}

std::expected<std::tuple<std::shared_ptr<CameraDataset>, torch::Tensor>, std::string>
BlenderReader::create_dataset() const {
    return create_dataset_from_transforms(m_datasetConfig);
}

PointCloud BlenderReader::createPointCloud() const {
    return generate_random_point_cloud();
}

bool BlenderReader::isValid() const {
    std::filesystem::path transformsFile = m_datasetConfig.data_path;
    if (std::filesystem::is_directory(m_datasetConfig.data_path)) {
        if (std::filesystem::is_regular_file(m_datasetConfig.data_path / "transforms_train.json")) {
            transformsFile = m_datasetConfig.data_path / "transforms_train.json";
        } else if (std::filesystem::is_regular_file(m_datasetConfig.data_path / "transforms.json")) {
            transformsFile = m_datasetConfig.data_path / "transforms_train.json";
        } else {
            std::println("could not find transforms_train.json nor transforms.json in: {}", transformsFile.string());
            return true;
        }
    }
    if (!std::filesystem::is_regular_file(transformsFile)) {
        // std::print(transformsFile.string()+" is not a file");
        return false;
    }

    return true;
}

// factory method implementation
std::unique_ptr<DataReader> GetValidDataReader(const gs::param::DatasetConfig& datasetConfig) {
    // Iterate through all DataReaderType enum values
    for (int enumValue = 0; enumValue <= 1; ++enumValue) {
        DataReaderType readerType = static_cast<DataReaderType>(enumValue);

        std::unique_ptr<DataReader> reader = nullptr;

        // Create appropriate reader based on enum value
        switch (readerType) {
        case DataReaderType::Colmap:
            reader = std::make_unique<ColmapReader>(datasetConfig);
            break;
        case DataReaderType::Blender:
            reader = std::make_unique<BlenderReader>(datasetConfig);
            break;
        }

        // Check if the reader is valid
        if (reader && reader->isValid()) {
            // Print the found valid reader type
            switch (readerType) {
            case DataReaderType::Colmap:
                std::println("found Colmap dataset");
                break;
            case DataReaderType::Blender:
                std::println("found Blender dataset");
                break;
            }
            return reader;
        } else {
            switch (readerType) {
            case DataReaderType::Colmap:
                std::println("Tried Colmap dataset and failed");
                break;
            case DataReaderType::Blender:
                std::println("Tried Colmap dataset and failed");
                break;
            }
        }
    }

    // No valid reader found - throw runtime exception
    throw std::runtime_error("No valid DataReader found for the given dataset configuration");
}