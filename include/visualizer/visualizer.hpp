/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <expected>
#include <filesystem>
#include <memory>
#include <string>

namespace gs {
    namespace param {
        struct TrainingParameters;
    }
} // namespace gs

namespace gs::management {
    class Project;
}

namespace gs::visualizer {

    struct ViewerOptions {
        std::string title = "LichtFeld Studio";
        int width = 1280;
        int height = 720;
        bool antialiasing = false;
        bool enable_cuda_interop = true;
        bool gut = false;
    };

    class Visualizer {
    public:
        static std::unique_ptr<Visualizer> create(const ViewerOptions& options = {});

        virtual void run() = 0;
        virtual void setParameters(const param::TrainingParameters& params) = 0;
        virtual std::expected<void, std::string> loadPLY(const std::filesystem::path& path) = 0;
        virtual std::expected<void, std::string> loadDataset(const std::filesystem::path& path) = 0;

        // project handling
        virtual bool openProject(const std::filesystem::path& path) = 0;
        virtual bool closeProject(const std::filesystem::path& path) = 0;
        virtual std::shared_ptr<gs::management::Project> getProject() = 0;
        virtual void attachProject(std::shared_ptr<gs::management::Project> _project) = 0;

        virtual void clearScene() = 0;

        virtual ~Visualizer() = default;
    };

} // namespace gs::visualizer
