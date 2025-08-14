#include "management/project.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace gs::management {

    // Static member definitions
    const Version LichtFeldProjectFile::CURRENT_VERSION(0, 0, 1);
    const std::string LichtFeldProjectFile::FILE_HEADER = "# LichtFeld Project File";

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

    YAML::Node MigratorRegistry::migrateToVersion(const YAML::Node& data, const Version& from, const Version& to) const {
        YAML::Node current = YAML::Clone(data);
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

    // LichtFeldProjectFile implementation
    LichtFeldProjectFile::LichtFeldProjectFile() {
        project_data_.version = CURRENT_VERSION;
        project_data_.project_creation_time = generateCreationTimeStamp();
        initializeMigrators();
    }

    LichtFeldProjectFile::LichtFeldProjectFile(const ProjectData& initialData)
        : project_data_(initialData) {
        initializeMigrators();
    }

    void LichtFeldProjectFile::initializeMigrators() {
        // Register migration classes for future versions
        // Example: migrator_registry_.registerMigrator(std::make_unique<Version001To002Migrator>());
    }

    bool LichtFeldProjectFile::readFromFile(const std::string& filepath) {
        try {
            YAML::Node doc = YAML::LoadFile(filepath);

            if (!validateYamlStructure(doc)) {
                std::cerr << "Invalid YAML structure in file: " << filepath << std::endl;
                return false;
            }

            // Check version and migrate if necessary
            Version fileVersion(doc["version"].as<std::string>());

            YAML::Node processedDoc = doc;
            if (fileVersion < CURRENT_VERSION) {
                std::cout << "Migrating from version " << fileVersion.toString()
                          << " to " << CURRENT_VERSION.toString() << std::endl;
                processedDoc = migrator_registry_.migrateToVersion(doc, fileVersion, CURRENT_VERSION);
            }

            project_data_ = parseProjectData(processedDoc);
            return true;

        } catch (const std::exception& e) {
            std::cerr << "Error reading project file: " << e.what() << std::endl;
            return false;
        }
    }

    bool LichtFeldProjectFile::writeToFile(const std::string& filepath) const {
        try {
            std::ofstream file(filepath);
            if (!file.is_open()) {
                std::cerr << "Cannot open file for writing: " << filepath << std::endl;
                return false;
            }

            // Write header comment
            file << FILE_HEADER << std::endl;

            // Serialize and write YAML
            YAML::Node doc = serializeProjectData(project_data_);
            file << doc << std::endl;

            return true;

        } catch (const std::exception& e) {
            std::cerr << "Error writing project file: " << e.what() << std::endl;
            return false;
        }
    }

    bool LichtFeldProjectFile::validateYamlStructure(const YAML::Node& node) const {
        // Basic validation - check required fields
        return node["version"] &&
               node["project_name"] &&
               node["project_creation_time"] &&
               node["data"] &&
               node["outputs"];
    }

    ProjectData LichtFeldProjectFile::parseProjectData(const YAML::Node& node) const {
        ProjectData data;

        data.version = Version(node["version"].as<std::string>());
        data.project_name = node["project_name"].as<std::string>();
        data.project_creation_time = node["project_creation_time"].as<std::string>();

        // Parse data section
        const auto& dataNode = node["data"];
        data.data.data_path = dataNode["data_path"].as<std::string>();
        data.data.data_type = dataNode["data_type"].as<std::string>();

        // Parse outputs section
        const auto& outputsNode = node["outputs"];
        if (outputsNode["plys"]) {
            for (const auto& plyNode : outputsNode["plys"]) {
                const auto& ply = plyNode["ply"];
                PlyData plyData;
                plyData.is_imported = ply["is_imported"].as<bool>();
                plyData.ply_path = ply["ply_path"].as<std::string>();
                plyData.ply_training_iter_number = ply["ply_training_iter_number"].as<int>();
                data.outputs.plys.push_back(plyData);
            }
        }

        return data;
    }

    YAML::Node LichtFeldProjectFile::serializeProjectData(const ProjectData& data) const {
        YAML::Node doc;

        doc["version"] = data.version.toString();
        doc["project_name"] = data.project_name;
        doc["project_creation_time"] = data.project_creation_time;

        // Data section
        doc["data"]["data_path"] = data.data.data_path;
        doc["data"]["data_type"] = data.data.data_type;

        // Outputs section
        for (const auto& ply : data.outputs.plys) {
            YAML::Node plyNode;
            plyNode["ply"]["is_imported"] = ply.is_imported;
            plyNode["ply"]["ply_path"] = ply.ply_path;
            plyNode["ply"]["ply_training_iter_number"] = ply.ply_training_iter_number;
            doc["outputs"]["plys"].push_back(plyNode);
        }

        return doc;
    }

    std::string LichtFeldProjectFile::generateCreationTimeStamp() const {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);

        std::stringstream ss;
        ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
        return ss.str();
    }

    // Convenience methods
    void LichtFeldProjectFile::setProjectName(const std::string& name) {
        project_data_.project_name = name;
    }

    void LichtFeldProjectFile::setDataInfo(const std::string& path, const std::string& type) {
        project_data_.data.data_path = path;
        project_data_.data.data_type = type;
    }

    void LichtFeldProjectFile::addPly(const PlyData& ply) {
        project_data_.outputs.plys.push_back(ply);
    }

    void LichtFeldProjectFile::removePly(size_t index) {
        if (index < project_data_.outputs.plys.size()) {
            project_data_.outputs.plys.erase(project_data_.outputs.plys.begin() + index);
        }
    }

    bool LichtFeldProjectFile::isCompatible(const Version& fileVersion) const {
        return fileVersion <= CURRENT_VERSION;
    }

    bool LichtFeldProjectFile::validateProjectData() const {
        return !project_data_.project_name.empty() &&
               !project_data_.data.data_path.empty() &&
               !project_data_.data.data_type.empty();
    }

} // namespace gs::management