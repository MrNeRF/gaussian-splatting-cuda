#pragma once

#include <expected>
#include <memory>
#include <string>
#include <torch/torch.h>
#include <tuple>

#include "core/camera.hpp"
#include "core/parameters.hpp"

enum class DataReaderType {
    Colmap = 0, ///< COLMAP dataset format
    Blender = 1 ///< Blender dataset format
};

// Forward declarations - adjust these includes based on your project structure
class CameraDataset;
class PointCloud;

/**
 * Abstract base class for reading different types of datasets
 */
class IDataReader {
public:
    /**
     * Constructor
     * @param datasetConfig Configuration parameters for the dataset
     */
    explicit IDataReader(const gs::param::DatasetConfig& datasetConfig);

    /**
     * Virtual destructor for proper cleanup of derived classes
     */
    virtual ~IDataReader() = default;

    // Make class non-copyable
    IDataReader(const IDataReader&) = delete;
    IDataReader& operator=(const IDataReader&) = delete;

    // Allow move operations (optional - remove if you want to disable moving too)
    IDataReader(IDataReader&&) = default;
    IDataReader& operator=(IDataReader&&) = default;

    /**
     * Creates a dataset with camera data and associated tensor
     * @return Expected containing tuple of CameraDataset and torch::Tensor on success,
     *         or error string on failure
     */
    virtual std::expected<std::tuple<std::shared_ptr<CameraDataset>, torch::Tensor>, std::string>
    create_dataset() const = 0;

    /**
     * Creates a point cloud from the dataset
     * @return PointCloud object
     */
    virtual PointCloud createPointCloud() const = 0;

    /**
     * Validates if the dataset configuration and data are valid
     * @return true if the dataset is valid and can be processed, false otherwise
     */
    virtual bool isValid() const = 0;

protected:
    /**
     * Store the dataset configuration for use by derived classes
     */
    const gs::param::DatasetConfig& m_datasetConfig;
};

// Constructor implementation
inline IDataReader::IDataReader(const gs::param::DatasetConfig& datasetConfig)
    : m_datasetConfig(datasetConfig) {
}

/**
 * BlenderReader - Derived class for reading Blender-generated datasets
 */
class BlenderReader : public IDataReader {
public:
    /**
     * Constructor - inherits from DataReader
     * @param datasetConfig Configuration parameters for the Blender dataset
     */
    explicit BlenderReader(const gs::param::DatasetConfig& datasetConfig);

    /**
     * Creates a dataset from Blender data format
     * @return Expected containing tuple of CameraDataset and torch::Tensor on success,
     *         or error string on failure
     */
    std::expected<std::tuple<std::shared_ptr<CameraDataset>, torch::Tensor>, std::string>
    create_dataset() const override;

    /**
     * Creates a point cloud from Blender dataset
     * @return PointCloud object created from Blender data
     */
    PointCloud createPointCloud() const override;

    /**
     * Validates if the dataset configuration and data are valid
     * @return true if the dataset is valid and can be processed, false otherwise
     */
    bool isValid() const override;
};

/**
 * ColmapReader - Derived class for reading COLMAP datasets
 */
class ColmapReader : public IDataReader {
public:
    /**
     * Constructor - inherits from DataReader
     * @param datasetConfig Configuration parameters for the COLMAP dataset
     */
    explicit ColmapReader(const gs::param::DatasetConfig& datasetConfig);

    /**
     * Creates a dataset from COLMAP data format
     * @return Expected containing tuple of CameraDataset and torch::Tensor on success,
     *         or error string on failure
     */
    std::expected<std::tuple<std::shared_ptr<CameraDataset>, torch::Tensor>, std::string>
    create_dataset() const override;

    /**
     * Creates a point cloud from COLMAP dataset
     * @return PointCloud object created from COLMAP data
     */
    PointCloud createPointCloud() const override;

    /**
     * Validates if the dataset configuration and data are valid
     * @return true if the dataset is valid and can be processed, false otherwise
     */
    bool isValid() const override;
};

// Inline constructor implementations
inline BlenderReader::BlenderReader(const gs::param::DatasetConfig& datasetConfig)
    : IDataReader(datasetConfig) {
}

inline ColmapReader::ColmapReader(const gs::param::DatasetConfig& datasetConfig)
    : IDataReader(datasetConfig) {
}

// factory method implementation
std::unique_ptr<IDataReader> GetValidDataReader(const gs::param::DatasetConfig& datasetConfig);
