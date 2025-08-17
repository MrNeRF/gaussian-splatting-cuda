#pragma once

#include <filesystem>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "core/parameters.hpp"

namespace gs::management {

    // Version structure for semantic versioning
    struct Version {
        int major;
        int minor;
        int patch;

        Version(int maj = 0, int min = 0, int p = 1) : major(maj),
                                                       minor(min),
                                                       patch(p) {}
        Version(const std::string& versionStr);

        std::string toString() const;
        bool operator>=(const Version& other) const;
        bool operator<(const Version& other) const;
        bool operator<=(const Version& other) const;
        bool operator>(const Version& other) const;
        bool operator==(const Version& other) const;
        bool operator!=(const Version& other) const;
    };

    // Data structures
    struct PlyData {
        bool is_imported = false;
        std::filesystem::path ply_path;
        int ply_training_iter_number = 0;

        // Constructor for easy initialization
        PlyData() = default;
        PlyData(bool imported, const std::filesystem::path& path, int iter)
            : is_imported(imported),
              ply_path(path),
              ply_training_iter_number(iter) {}
    };

    struct DataInfo {
        std::string data_path;
        std::string data_type;

        DataInfo() = default;
        DataInfo(const std::string& path, const std::string& type)
            : data_path(path),
              data_type(type) {}
    };

    struct OutputsInfo {
        std::vector<PlyData> plys;
    };

    // Main project data structure
    struct ProjectData {
        Version version;
        std::string project_name;
        std::string project_creation_time;
        std::string project_last_update_time;
        DataInfo data;
        OutputsInfo outputs;

        // Additional fields for future versions can be added here
        nlohmann::json additional_fields; // For storing unknown fields during migration
    };

    // Migration interface for backward compatibility
    class ProjectMigrator {
    public:
        virtual ~ProjectMigrator() = default;
        virtual bool canMigrate(const Version& from, const Version& to) const = 0;
        virtual nlohmann::json migrate(const nlohmann::json& oldData, const Version& from, const Version& to) const = 0;
    };

    // Concrete migrator implementations
    class MigratorRegistry {
    private:
        std::vector<std::unique_ptr<ProjectMigrator>> migrators_;

    public:
        void registerMigrator(std::unique_ptr<ProjectMigrator> migrator);
        nlohmann::json migrateToVersion(const nlohmann::json& data, const Version& from, const Version& to) const;
    };

    // Main project file manager class
    class LichtFeldProject {
    private:
        ProjectData project_data_;
        MigratorRegistry migrator_registry_;

        // Internal helper methods
        void initializeMigrators();
        bool validateJsonStructure(const nlohmann::json& json) const;
        ProjectData parseProjectData(const nlohmann::json& json) const;
        nlohmann::json serializeProjectData(const ProjectData& data) const;

    public:
        static const Version CURRENT_VERSION;
        static const std::string FILE_HEADER;
        static const std::string EXTENSION;

        LichtFeldProject(bool update_file_on_change = false);
        explicit LichtFeldProject(const ProjectData& initialData);

        void setOutputFileName(const std::filesystem::path& path);
        std::filesystem::path getOutputPath() const { return output_file_name_; }

        // Main interface methods
        bool readFromFile(const std::filesystem::path& filepath);
        // if the user gave a path - use path else use the one that was given in setOutputFileName
        bool writeToFile(const std::filesystem::path& filepath = {});

        // Data access methods
        const ProjectData& getProjectData() const { return project_data_; }
        ProjectData& getProjectData() { return project_data_; }

        void setProjectData(const ProjectData& data) { project_data_ = data; }

        // Convenience methods
        void setProjectName(const std::string& name);
        void setDataInfo(const std::filesystem::path& path, const std::string& type);
        // detect type automatically
        void setDataInfo(const std::filesystem::path& path);
        void addPly(const PlyData& ply);
        void removePly(size_t index);

        // Version and compatibility methods
        Version getFileVersion() const { return project_data_.version; }
        static Version getCurrentVersion() { return CURRENT_VERSION; }
        bool isCompatible(const Version& fileVersion) const;

        // Utility methods
        std::string generateCurrentTimeStamp() const;
        bool validateProjectData() const;

    private:
        std::filesystem::path output_file_name_;
        bool update_file_on_change_ = false; // if true update file on every change
        mutable std::mutex io_mutex_;
    };

    std::shared_ptr<LichtFeldProject> GetLichtFeldProject(const gs::param::DatasetConfig& data,
                                                          const std::string& project_name = "LichtFeldStudioProject");

} // namespace gs::management