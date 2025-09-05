/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "core/logger.hpp"
#include "project/project.hpp"

namespace gs::management {

    // Static member definitions
    const Version Project::CURRENT_VERSION(0, 0, 1);
    const std::string Project::FILE_HEADER = "LichtFeldStudio Project File";
    const std::string Project::EXTENSION = ".ls"; // LichtFeldStudio file

    // Version implementation
    Version::Version(const std::string& versionStr) {
        std::istringstream ss(versionStr);
        std::string token;

        std::getline(ss, token, '.');
        major = std::stoi(token);

        std::getline(ss, token, '.');
        minor = std::stoi(token);

        std::getline(ss, token);
        patch = std::stoi(token);
    }

    std::string Version::toString() const {
        return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
    }

    bool Version::operator>=(const Version& other) const {
        if (major != other.major)
            return major > other.major;
        if (minor != other.minor)
            return minor > other.minor;
        return patch >= other.patch;
    }

    bool Version::operator<(const Version& other) const {
        return !(*this >= other);
    }

    bool Version::operator<=(const Version& other) const {
        return *this < other || *this == other;
    }

    bool Version::operator>(const Version& other) const {
        return !(*this <= other);
    }

    bool Version::operator==(const Version& other) const {
        return major == other.major && minor == other.minor && patch == other.patch;
    }

    bool Version::operator!=(const Version& other) const {
        return !(*this == other);
    }

    // MigratorRegistry implementation
    void MigratorRegistry::registerMigrator(std::unique_ptr<ProjectMigrator> migrator) {
        migrators_.push_back(std::move(migrator));
    }

    nlohmann::json MigratorRegistry::migrateToVersion(const nlohmann::json& data, const Version& from, const Version& to) const {
        nlohmann::json current = data;
        Version currentVersion = from;

        while (currentVersion < to) {
            bool migrationFound = false;
            for (const auto& migrator : migrators_) {
                if (migrator->canMigrate(currentVersion, to)) {
                    current = migrator->migrate(current, currentVersion, to);
                    currentVersion = to; // For now, assume direct migration
                    migrationFound = true;
                    break;
                }
            }

            if (!migrationFound) {
                throw std::runtime_error("No migration path found from version " +
                                         currentVersion.toString() + " to " + to.toString());
            }
        }

        return current;
    }

    bool IsColmapData(const std::filesystem::path& path) {
        if (!std::filesystem::is_directory(path)) {
            return false;
        }
        // Check for sparse reconstruction
        std::filesystem::path sparse_path;
        if (std::filesystem::exists(path / "sparse" / "0")) {
            sparse_path = path / "sparse" / "0";
        } else if (std::filesystem::exists(path / "sparse")) {
            sparse_path = path / "sparse";
        } else {
            return false;
        }

        return true;
    }

    DataSetInfo::DataSetInfo(const param::DatasetConfig& data_config) : DatasetConfig(data_config) {
        if (data_path.empty()) {
            data_type = "";
        } else {
            data_type = IsColmapData(data_path) ? "Colmap" : "Blender";
        }
    }

    // LichtFeldProject implementation
    Project::Project(bool update_file_on_change) : update_file_on_change_(update_file_on_change) {
        project_data_.version = CURRENT_VERSION;
        project_data_.project_creation_time = generateCurrentTimeStamp();
        initializeMigrators();

        if (update_file_on_change_ && !output_file_name_.empty()) {
            writeToFile();
        }
    }

    void Project::setProjectFileName(const std::filesystem::path& path) {
        if (std::filesystem::is_directory(path)) {
            std::string project_file_name = project_data_.project_name.empty() ? "project" : project_data_.project_name;
            project_file_name += EXTENSION;
            output_file_name_ = path / project_file_name;
        } else if (std::filesystem::is_regular_file(path)) {
            if (path.extension() != EXTENSION) {
                throw std::runtime_error(std::format("LichtFeldProjectFile: {} expected file extension to be {}", path.string(), EXTENSION));
            }
        }
        output_file_name_ = path;
    }

    Project::Project(const ProjectData& initialData, bool update_file_on_change)
        : project_data_(initialData),
          update_file_on_change_(update_file_on_change) {
        initializeMigrators();
    }

    void Project::initializeMigrators() {
        // Register migration classes for future versions
        // Example: migrator_registry_.registerMigrator(std::make_unique<Version001To002Migrator>());
    }

    bool Project::readFromFile(const std::filesystem::path& filepath) {
        std::lock_guard<std::mutex> lock(io_mutex_);
        try {
            std::ifstream file(filepath);
            if (!file.is_open()) {
                LOG_ERROR("Cannot open file for reading: {}", filepath.string());
                return false;
            }

            nlohmann::json doc;
            file >> doc;

            if (!validateJsonStructure(doc)) {
                LOG_ERROR("Invalid JSON structure in file: {}", filepath.string());
                return false;
            }

            // Check version and migrate if necessary
            Version fileVersion(doc["version"].get<std::string>());

            nlohmann::json processedDoc = doc;
            if (fileVersion < CURRENT_VERSION) {
                LOG_INFO("Migrating from version {} to {}", fileVersion.toString(), CURRENT_VERSION.toString());
                processedDoc = migrator_registry_.migrateToVersion(doc, fileVersion, CURRENT_VERSION);
            }

            project_data_ = parseProjectData(processedDoc);
            output_file_name_ = filepath;

            return true;

        } catch (const std::exception& e) {
            LOG_ERROR("Error reading project file: {}", e.what());
            return false;
        }
    }

    bool Project::writeToFile(const std::filesystem::path& filepath) {
        std::lock_guard<std::mutex> lock(io_mutex_);

        std::filesystem::path targetPath = filepath.empty() ? output_file_name_ : filepath;
        if (targetPath.empty()) {
            LOG_ERROR("LichtFeldProjectFile::writeToFile - no output file was set");
            return false;
        }

        if (std::filesystem::is_directory(targetPath)) {
            LOG_ERROR("LichtFeldProjectFile: {} is directory and not a file", targetPath.string());
            return false;
        }

        if (!std::filesystem::is_directory(targetPath.parent_path())) {
            LOG_ERROR("LichtFeldProjectFile: {} parent directory does not exist {}", targetPath.parent_path().string(), targetPath.string());
            return false;
        }

        if (targetPath.extension() != EXTENSION) {
            LOG_ERROR("LichtFeldProjectFile: {} expected file extension to be {}", targetPath.string(), EXTENSION);
            return false;
        }

        project_data_.project_last_update_time = generateCurrentTimeStamp();

        try {
            std::ofstream file(targetPath);
            if (!file.is_open()) {
                LOG_ERROR("Cannot open file for writing: {}", targetPath.string());
                return false;
            }

            // Serialize and write JSON
            nlohmann::ordered_json doc = serializeProjectData(project_data_);
            file << doc.dump(4) << std::endl; // Pretty print with 4-space indentation

            return true;

        } catch (const std::exception& e) {
            LOG_ERROR("Error writing project file: {}", e.what());
            return false;
        }
    }

    bool Project::validateJsonStructure(const nlohmann::json& json) const {
        // Basic validation - check required fields
        bool contains_basics = json.contains("project_info") &&
                               json.contains("version") &&
                               json.contains("project_name") &&
                               json.contains("project_creation_time") &&
                               json.contains("project_last_update_time") &&
                               json.contains("project_output_folder") &&
                               json.contains("data") &&
                               json.contains("outputs");
        if (!contains_basics) {
            return false;
        }

        const auto& dataJson = json["data"];
        bool contains_data = dataJson.contains("data_path") &&
                             dataJson.contains("images") &&
                             dataJson.contains("resize_factor") &&
                             dataJson.contains("test_every") &&
                             dataJson.contains("data_type");

        return contains_data;
    }

    ProjectData Project::parseProjectData(const nlohmann::json& json) const {
        ProjectData data;

        // Parse version
        if (json.contains("version")) {
            data.version = Version(json["version"].get<std::string>());
        }

        // Parse metadata
        if (json.contains("project_name")) {
            data.project_name = json["project_name"].get<std::string>();
        }
        if (json.contains("project_creation_time")) {
            data.project_creation_time = json["project_creation_time"].get<std::string>();
        }
        if (json.contains("project_last_update_time")) {
            data.project_last_update_time = json["project_last_update_time"].get<std::string>();
        }

        // Parse dataset info
        if (json.contains("data_set_info")) {
            const auto& ds = json["data_set_info"];
            if (ds.contains("data_path")) {
                data.data_set_info.data_path = ds["data_path"].get<std::string>();
            }
            if (ds.contains("output_path")) {
                data.data_set_info.output_path = ds["output_path"].get<std::string>();
            }
            if (ds.contains("project_path")) {
                data.data_set_info.project_path = ds["project_path"].get<std::string>();
            }
            if (ds.contains("images")) {
                data.data_set_info.images = ds["images"].get<std::string>();
            }
            if (ds.contains("resize_factor")) {
                data.data_set_info.resize_factor = ds["resize_factor"].get<int>();
            }
            if (ds.contains("test_every")) {
                data.data_set_info.test_every = ds["test_every"].get<int>();
            }
            if (ds.contains("data_type")) {
                data.data_set_info.data_type = ds["data_type"].get<std::string>();
            }
        }

        if (json.contains("training") && json["training"].contains("optimization")) {
            // The optimization JSON should contain a "strategy" field
            const auto& opt_json = json["training"]["optimization"];
            std::string strategy = opt_json.value("strategy", "default");

            // Load the base parameters for the strategy
            auto opt_params_result = param::read_optim_params_from_json(strategy);
            if (opt_params_result) {
                // Apply overrides from the saved JSON
                data.optimization.params = opt_params_result->params.with_overrides(opt_json);
                data.optimization.strategy = strategy;
            } else {
                // Fallback: create default parameters
                data.optimization = param::OptimizationParameters();
                data.optimization.strategy = strategy;
            }
        }

        // Parse outputs
        if (json.contains("outputs") && json["outputs"].contains("plys")) {
            for (const auto& ply : json["outputs"]["plys"]) {
                PlyData ply_data;
                if (ply.contains("is_imported")) {
                    ply_data.is_imported = ply["is_imported"].get<bool>();
                }
                if (ply.contains("ply_path")) {
                    ply_data.ply_path = ply["ply_path"].get<std::string>();
                }
                if (ply.contains("ply_training_iter_number")) {
                    ply_data.ply_training_iter_number = ply["ply_training_iter_number"].get<int>();
                }
                if (ply.contains("ply_name")) {
                    ply_data.ply_name = ply["ply_name"].get<std::string>();
                }
                data.outputs.plys.push_back(ply_data);
            }
        }

        // Store any additional fields for future compatibility
        for (auto& [key, value] : json.items()) {
            if (key != "version" && key != "project_name" && key != "project_creation_time" &&
                key != "project_last_update_time" && key != "data_set_info" &&
                key != "training" && key != "outputs") {
                data.additional_fields[key] = value;
            }
        }

        return data;
    }

    nlohmann::ordered_json Project::serializeProjectData(const ProjectData& data) const {
        nlohmann::ordered_json json;

        // Add project info as the first field
        json["project_info"] = FILE_HEADER;
        json["version"] = data.version.toString();
        json["project_name"] = data.project_name;
        json["project_creation_time"] = data.project_creation_time;
        json["project_last_update_time"] = data.project_last_update_time;
        json["project_output_folder"] = data.data_set_info.output_path;

        // Data section
        json["data"]["data_path"] = data.data_set_info.data_path;
        json["data"]["data_type"] = data.data_set_info.data_type;
        json["data"]["resize_factor"] = data.data_set_info.resize_factor;
        json["data"]["test_every"] = data.data_set_info.test_every;
        json["data"]["images"] = data.data_set_info.images;

        // training optimization
        json["training"]["optimization"] = data.optimization.to_json();

        // Outputs section
        json["outputs"]["plys"] = nlohmann::ordered_json::array();
        for (const auto& ply : data.outputs.plys) {
            nlohmann::ordered_json plyJson;
            plyJson["is_imported"] = ply.is_imported;
            plyJson["ply_path"] = ply.ply_path.string();
            plyJson["ply_training_iter_number"] = ply.ply_training_iter_number;
            plyJson["ply_name"] = ply.ply_name;
            json["outputs"]["plys"].push_back(plyJson);
        }

        // Merge any additional fields
        if (!data.additional_fields.empty()) {
            json.update(data.additional_fields);
        }

        return json;
    }

    std::string Project::generateCurrentTimeStamp() const {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);

        std::stringstream ss;
        ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
        return ss.str();
    }

    // Convenience methods
    void Project::setProjectName(const std::string& name) {
        project_data_.project_name = name;
    }

    void Project::setDataInfo(const param::DatasetConfig& data_config) {
        project_data_.data_set_info = DataSetInfo(data_config);
        std::string datatype = IsColmapData(project_data_.data_set_info.data_path) ? "Colmap" : "Blender";

        project_data_.data_set_info.data_type = datatype;

        if (update_file_on_change_ && !output_file_name_.empty()) {
            writeToFile();
        }
    }

    bool Project::addPly(const PlyData& ply_to_be_added) {

        for (const auto& ply : project_data_.outputs.plys) {
            if (ply.ply_name == ply_to_be_added.ply_name) {
                LOG_ERROR("can not insert two plys with the same name");
                return false;
            }
        }

        project_data_.outputs.plys.push_back(ply_to_be_added);

        if (update_file_on_change_ && !output_file_name_.empty()) {
            if (!writeToFile()) {
                return false;
            }
        }
        return true;
    }

    std::vector<PlyData> Project::getPlys() const {
        return project_data_.outputs.plys;
    }

    void Project::removePly(size_t index) {
        if (index < project_data_.outputs.plys.size()) {
            project_data_.outputs.plys.erase(project_data_.outputs.plys.begin() + index);
        }

        if (update_file_on_change_ && !output_file_name_.empty()) {
            writeToFile();
        }
    }

    void Project::clearPlys() {
        project_data_.outputs.plys.clear();
        if (update_file_on_change_ && !output_file_name_.empty()) {
            writeToFile();
        }
    }

    bool Project::isCompatible(const Version& fileVersion) const {
        return fileVersion <= CURRENT_VERSION;
    }

    bool Project::validateProjectData() const {
        return !project_data_.project_name.empty() &&
               !project_data_.data_set_info.data_path.empty() &&
               !project_data_.data_set_info.data_type.empty();
    }

    bool Project::portProjectToDir(const std::filesystem::path& dst_dir) {

        namespace fs = std::filesystem;

        if (!std::filesystem::is_directory(dst_dir)) {
            LOG_ERROR("PortProjectToDir: Directory does not exists {}", dst_dir.string());
            return false;
        }

        const fs::path src_project_dir = getProjectOutputFolder();

        setProjectOutputFolder(dst_dir);
        if (!fs::is_regular_file(output_file_name_)) {
            LOG_ERROR("PortProjectToDir: {} orig path does not exists", output_file_name_.string());
        }
        const std::string proj_filename = output_file_name_.filename().string();
        const fs::path dst_project_file_path = dst_dir / proj_filename;

        setProjectFileName(dst_project_file_path);

        // Copy all ply files into new directory
        for (auto& ply : project_data_.outputs.plys) {
            try {
                if (!fs::exists(ply.ply_path)) {
                    LOG_ERROR("PortProjectToDir: ply file does not exist: {}", ply.ply_path.string());
                    return false;
                }

                // Keep same filename, copy to dst_dir
                fs::path dst_ply_path = dst_dir / ply.ply_path.filename();

                // Overwrite if already exists
                fs::copy_file(ply.ply_path, dst_ply_path, fs::copy_options::overwrite_existing);

                // Update metadata path
                ply.ply_path = dst_ply_path;

            } catch (const fs::filesystem_error& e) {
                LOG_ERROR("PortProjectToDir: failed to copy ply {} -> {}. reason: {}",
                          ply.ply_path.string(), (dst_dir / ply.ply_path.filename()).string(), e.what());
                return false;
            }
        }

        // Finally, write updated project file
        if (!writeToFile()) {
            LOG_ERROR("PortProjectToDir: failed to write updated project file to {}", dst_project_file_path.string());
            return false;
        }

        LOG_INFO("Project was successfully ported to {}", dst_project_file_path.string());

        return true;
    }

    std::shared_ptr<Project> CreateNewProject(const gs::param::DatasetConfig& data,
                                              const param::OptimizationParameters& opt,
                                              const std::string& project_name,
                                              bool update_file_on_change) {

        auto project = std::make_shared<gs::management::Project>(update_file_on_change);

        project->setProjectName(project_name);
        if (data.output_path.empty()) {
            LOG_ERROR("output_path is empty");
            return nullptr;
        }
        std::filesystem::path project_path = data.project_path;
        if (project_path.empty()) {
            project_path = data.output_path / ("project" + Project::EXTENSION);
            LOG_INFO("project_path is empty - creating new project{} file", Project::EXTENSION);
        }

        if (project_path.extension() != Project::EXTENSION) {
            LOG_ERROR("project_path must be {} file: {}", Project::EXTENSION, project_path.string());
            return nullptr;
        }
        if (project_path.parent_path().empty()) {
            LOG_ERROR("project_path must have parent directory: project_path: {} ", project_path.string());
            return nullptr;
        }

        try {
            project->setProjectFileName(project_path);
            project->setProjectOutputFolder(data.output_path);
            project->setDataInfo(data);
            project->setOptimizationParams(opt);
        } catch (const std::exception& e) {
            LOG_ERROR("Error writing project file: {}", e.what());
            return nullptr;
        }

        return project;
    }

    std::filesystem::path FindProjectFile(const std::filesystem::path& directory) {
        if (!std::filesystem::exists(directory) || !std::filesystem::is_directory(directory)) {
            return {};
        }

        std::filesystem::path foundPath;
        int count = 0;

        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file() && entry.path().extension() == ".ls") {
                ++count;
                if (count == 1) {
                    foundPath = entry.path();
                } else {
                    LOG_ERROR("Multiple .ls files found in {}", directory.string());
                    return {};
                }
            }
        }

        if (count == 0) {
            return {};
        }
        return foundPath;
    }

    void clear_directory(const std::filesystem::path& path) {
        namespace fs = std::filesystem;
        for (const auto& entry : fs::directory_iterator(path)) {
            fs::remove_all(entry);
        }
    }

    std::shared_ptr<Project> CreateTempNewProject(const gs::param::DatasetConfig& data,
                                                  const param::OptimizationParameters& opt,
                                                  const std::string& project_name, bool udpdate_file_on_change) {
        namespace fs = std::filesystem;
        gs::param::DatasetConfig data_with_temp_output = data;

        auto temp_path = fs::temp_directory_path() / "LichtFeldStudio";
        if (fs::exists(temp_path)) {
            clear_directory(temp_path);
            LOG_INFO("Project temoprary directory exists removing its contenet {}", temp_path.string());
        } else {
            try {
                bool success = fs::create_directories(temp_path);
                if (!success) {
                    LOG_ERROR("failed to create temporary directory {}", temp_path.string());
                    return nullptr;
                }
                LOG_INFO("Project created temoprary directory successfuly: {}", temp_path.string());

            } catch (const fs::filesystem_error& e) {
                LOG_ERROR("failed to create temporary directory {}. reason: {}", temp_path.string(), e.what());
                return nullptr;
            }
        }
        data_with_temp_output.output_path = temp_path;
        auto project = CreateNewProject(data_with_temp_output, opt, project_name, udpdate_file_on_change);

        if (project) {
            project->setIsTempProject(true);
        }

        return project;
    }

} // namespace gs::management