#include "project/project.hpp"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <random>
#include <string>
#include <vector>

class ProjectTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary directory for test files
        temp_dir_ = std::filesystem::temp_directory_path() / "project_test";
        std::filesystem::create_directories(temp_dir_);

        // Setup random number generator
        rng_.seed(33550336);
    }

    void TearDown() override {
        // Clean up temporary directory
        if (std::filesystem::exists(temp_dir_)) {
            std::filesystem::remove_all(temp_dir_);
        }
    }

    std::string generateRandomString(size_t length) {
        const std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_";
        std::uniform_int_distribution<> dist(0, chars.size() - 1);

        std::string result;
        result.reserve(length);
        for (size_t i = 0; i < length; ++i) {
            result += chars[dist(rng_)];
        }
        return result;
    }

    std::filesystem::path generateRandomPath() {
        return std::filesystem::path("/") / generateRandomString(8) / generateRandomString(12);
    }

    gs::management::ProjectData generateRandomProjectData() {
        gs::management::ProjectData data;

        // Generate random basic info
        data.version = gs::management::Version(
            std::uniform_int_distribution<>(0, 5)(rng_),
            std::uniform_int_distribution<>(0, 20)(rng_),
            std::uniform_int_distribution<>(0, 100)(rng_));
        data.project_name = generateRandomString(15);
        data.project_creation_time = "2025-01-" + std::to_string(std::uniform_int_distribution<>(1, 28)(rng_)) + "T12:00:00Z";
        data.project_last_update_time = "2025-01-" + std::to_string(std::uniform_int_distribution<>(1, 28)(rng_)) + "T15:30:00Z";

        // Generate random data info
        data.data_set_info.data_path = generateRandomPath().string();
        std::vector<std::string> data_types = {"Colmap", "Blender", "Custom"};
        data.data_set_info.data_type = data_types[std::uniform_int_distribution<>(0, data_types.size() - 1)(rng_)];
        data.data_set_info.images = generateRandomString(8);
        data.data_set_info.resize_factor = std::uniform_int_distribution<>(0, 5)(rng_);
        data.data_set_info.test_every = std::uniform_int_distribution<>(0, 5)(rng_);

        // too lazy to test all fields - only test 2
        data.optimization.grad_threshold = std::uniform_real_distribution<float>(0, 10)(rng_);
        data.optimization.init_num_pts = std::uniform_int_distribution<>(0, 500000)(rng_);

        // Generate random PLY data
        int ply_count = std::uniform_int_distribution<>(0, 5)(rng_);
        for (int i = 0; i < ply_count; ++i) {
            gs::management::PlyData ply;
            ply.is_imported = std::uniform_int_distribution<>(0, 1)(rng_);
            ply.ply_path = generateRandomPath();
            ply.ply_training_iter_number = std::uniform_int_distribution<>(100, 10000)(rng_);
            ply.ply_name = generateRandomString(10);
            data.outputs.plys.push_back(ply);
        }

        return data;
    }

    bool compareProjectData(const gs::management::ProjectData& a, const gs::management::ProjectData& b) {
        // Compare version
        if (!(a.version == b.version)) {
            std::cout << "Version mismatch: " << a.version.toString() << " vs " << b.version.toString() << std::endl;
            return false;
        }

        // Compare strings
        if (a.project_name != b.project_name) {
            std::cout << "Project name mismatch: '" << a.project_name << "' vs '" << b.project_name << "'" << std::endl;
            return false;
        }

        if (a.project_creation_time != b.project_creation_time) {
            std::cout << "Creation time mismatch: '" << a.project_creation_time << "' vs '" << b.project_creation_time << "'" << std::endl;
            return false;
        }

        if (abs(a.optimization.grad_threshold - b.optimization.grad_threshold) > 1e-5) {
            std::cout << "grad_threshold mismatch: '" << a.optimization.grad_threshold << "' vs '" << b.optimization.grad_threshold << "'" << std::endl;
            return false;
        }

        if (a.optimization.init_num_pts != b.optimization.init_num_pts) {
            std::cout << "init_num_pts mismatch: '" << a.optimization.init_num_pts << "' vs '" << b.optimization.init_num_pts << "'" << std::endl;
            return false;
        }

        if (a.data_set_info.output_path != b.data_set_info.output_path) {
            std::cout << "project_output_folder mismatch: '" << a.data_set_info.output_path << "' vs '" << b.data_set_info.output_path << "'" << std::endl;
            return false;
        }

        if (a.data_set_info.images != b.data_set_info.images) {
            std::cout << "images mismatch: '" << a.data_set_info.images << "' vs '" << b.data_set_info.images << "'" << std::endl;
            return false;
        }

        if (a.data_set_info.resize_factor != b.data_set_info.resize_factor) {
            std::cout << "resize_factor mismatch: '" << a.data_set_info.resize_factor << "' vs '" << b.data_set_info.resize_factor << "'" << std::endl;
            return false;
        }

        if (a.data_set_info.test_every != b.data_set_info.test_every) {
            std::cout << "test_every mismatch: '" << a.data_set_info.test_every << "' vs '" << b.data_set_info.test_every << "'" << std::endl;
            return false;
        }

        // Compare data info
        if (a.data_set_info.data_path != b.data_set_info.data_path) {
            std::cout << "Data path mismatch: '" << a.data_set_info.data_path << "' vs '" << b.data_set_info.data_path << "'" << std::endl;
            return false;
        }

        if (a.data_set_info.data_type != b.data_set_info.data_type) {
            std::cout << "Data type mismatch: '" << a.data_set_info.data_type << "' vs '" << b.data_set_info.data_type << "'" << std::endl;
            return false;
        }

        // Compare PLY outputs
        if (a.outputs.plys.size() != b.outputs.plys.size()) {
            std::cout << "PLY count mismatch: " << a.outputs.plys.size() << " vs " << b.outputs.plys.size() << std::endl;
            return false;
        }

        for (size_t i = 0; i < a.outputs.plys.size(); ++i) {
            const auto& ply_a = a.outputs.plys[i];
            const auto& ply_b = b.outputs.plys[i];

            if (ply_a.is_imported != ply_b.is_imported) {
                std::cout << "PLY[" << i << "] is_imported mismatch: " << ply_a.is_imported << " vs " << ply_b.is_imported << std::endl;
                return false;
            }

            if (ply_a.ply_path != ply_b.ply_path) {
                std::cout << "PLY[" << i << "] path mismatch: '" << ply_a.ply_path << "' vs '" << ply_b.ply_path << "'" << std::endl;
                return false;
            }

            if (ply_a.ply_training_iter_number != ply_b.ply_training_iter_number) {
                std::cout << "PLY[" << i << "] iter number mismatch: " << ply_a.ply_training_iter_number << " vs " << ply_b.ply_training_iter_number << std::endl;
                return false;
            }

            if (ply_a.ply_name != ply_b.ply_name) {
                std::cout << "PLY[" << i << "] name mismatch: '" << ply_a.ply_name << "' vs '" << ply_b.ply_name << "'" << std::endl;
                return false;
            }
        }

        return true;
    }

    std::filesystem::path temp_dir_;
    std::mt19937 rng_;
};

TEST_F(ProjectTest, WriteReadCompareRandomProject) {
    // Generate random project data
    auto original_data = generateRandomProjectData();

    // Create temporary file path
    std::filesystem::path temp_file = temp_dir_ / (generateRandomString(10) + gs::management::Project::EXTENSION);

    // Create project with the random data
    gs::management::Project project(original_data);
    project.setProjectFileName(temp_file);

    // Write to file
    ASSERT_TRUE(project.writeToFile()) << "Failed to write project to file: " << temp_file;

    // Verify file was created
    ASSERT_TRUE(std::filesystem::exists(temp_file)) << "Project file was not created: " << temp_file;

    // Create a new project instance and read from file
    gs::management::Project loaded_project(false);
    ASSERT_TRUE(loaded_project.readFromFile(temp_file)) << "Failed to read project from file: " << temp_file;

    // Compare the data
    const auto& loaded_data = loaded_project.getProjectData();
    EXPECT_TRUE(compareProjectData(original_data, loaded_data)) << "Original and loaded project data do not match";

    // Verify file structure (check if it's valid JSON and has expected fields)
    std::ifstream file(temp_file);
    ASSERT_TRUE(file.is_open()) << "Cannot open written file for verification";

    nlohmann::json json_content;
    file >> json_content;

    // Check required fields exist
    EXPECT_TRUE(json_content.contains("project_info")) << "Missing project_info field";
    EXPECT_TRUE(json_content.contains("version")) << "Missing version field";
    EXPECT_TRUE(json_content.contains("project_name")) << "Missing project_name field";
    EXPECT_TRUE(json_content.contains("data")) << "Missing data field";
    EXPECT_TRUE(json_content.contains("outputs")) << "Missing outputs field";

    // Verify project_info content
    EXPECT_EQ(json_content["project_info"].get<std::string>(), "LichtFeldStudio Project File");

    // Clean up is handled by TearDown()
}

TEST_F(ProjectTest, MultipleRandomProjects) {
    // Test multiple random projects to increase confidence
    for (int test_case = 0; test_case < 10; ++test_case) {
        SCOPED_TRACE("Test case: " + std::to_string(test_case));

        auto original_data = generateRandomProjectData();
        std::filesystem::path temp_file = temp_dir_ / ("test_" + std::to_string(test_case) + gs::management::Project::EXTENSION);

        gs::management::Project project(original_data);
        project.setProjectFileName(temp_file);

        ASSERT_TRUE(project.writeToFile());
        ASSERT_TRUE(std::filesystem::exists(temp_file));

        gs::management::Project loaded_project(false);
        ASSERT_TRUE(loaded_project.readFromFile(temp_file));

        const auto& loaded_data = loaded_project.getProjectData();
        EXPECT_TRUE(compareProjectData(original_data, loaded_data));
    }
}

TEST_F(ProjectTest, EmptyPlyArrayHandling) {
    // Test edge case with empty PLY array
    auto data = generateRandomProjectData();
    data.outputs.plys.clear(); // Ensure empty PLY array

    std::filesystem::path temp_file = temp_dir_ / ("empty_plys" + gs::management::Project::EXTENSION);

    gs::management::Project project(data);
    project.setProjectFileName(temp_file);

    ASSERT_TRUE(project.writeToFile());

    gs::management::Project loaded_project(false);
    ASSERT_TRUE(loaded_project.readFromFile(temp_file));

    const auto& loaded_data = loaded_project.getProjectData();
    EXPECT_TRUE(compareProjectData(data, loaded_data));
    EXPECT_TRUE(loaded_data.outputs.plys.empty());
}