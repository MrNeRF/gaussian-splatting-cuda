#pragma once

#include <memory>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

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
        std::string ply_path;
        int ply_training_iter_number = 0;

        // Constructor for easy initialization
        PlyData() = default;
        PlyData(bool imported, const std::string& path, int iter)
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
        DataInfo data;
        OutputsInfo outputs;

        // Additional fields for future versions can be added here
        YAML::Node additional_fields; // For storing unknown fields during migration
    };

    // Migration interface for backward compatibility
    class ProjectMigrator {
    public:
        virtual ~ProjectMigrator() = default;
        virtual bool canMigrate(const Version& from, const Version& to) const = 0;
        virtual YAML::Node migrate(const YAML::Node& oldData, const Version& from, const Version& to) const = 0;
    };

    // Concrete migrator implementations
    class MigratorRegistry {
    private:
        std::vector<std::unique_ptr<ProjectMigrator>> migrators_;

    public:
        void registerMigrator(std::unique_ptr<ProjectMigrator> migrator);
        YAML::Node migrateToVersion(const YAML::Node& data, const Version& from, const Version& to) const;
    };

    // Main project file manager class
    class LichtFeldProjectFile {
    private:
        static const Version CURRENT_VERSION;
        static const std::string FILE_HEADER;

        ProjectData project_data_;
        MigratorRegistry migrator_registry_;

        // Internal helper methods
        void initializeMigrators();
        bool validateYamlStructure(const YAML::Node& node) const;
        ProjectData parseProjectData(const YAML::Node& node) const;
        YAML::Node serializeProjectData(const ProjectData& data) const;

    public:
        LichtFeldProjectFile();
        explicit LichtFeldProjectFile(const ProjectData& initialData);

        // Main interface methods
        bool readFromFile(const std::string& filepath);
        bool writeToFile(const std::string& filepath) const;

        // Data access methods
        const ProjectData& getProjectData() const { return project_data_; }
        ProjectData& getProjectData() { return project_data_; }

        void setProjectData(const ProjectData& data) { project_data_ = data; }

        // Convenience methods
        void setProjectName(const std::string& name);
        void setDataInfo(const std::string& path, const std::string& type);
        void addPly(const PlyData& ply);
        void removePly(size_t index);

        // Version and compatibility methods
        Version getFileVersion() const { return project_data_.version; }
        static Version getCurrentVersion() { return CURRENT_VERSION; }
        bool isCompatible(const Version& fileVersion) const;

        // Utility methods
        std::string generateCreationTimeStamp() const;
        bool validateProjectData() const;
    };

} // namespace gs::management