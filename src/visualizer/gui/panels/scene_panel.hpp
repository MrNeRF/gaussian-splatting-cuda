/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// Add these includes at the top of the file
#include <ShlObj.h>     // For BROWSEINFOA, SHBrowseForFolderA, etc.
#include <windows.h>    // For Windows types like LPITEMIDLIST
#include "core/events.hpp"
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "training/training_manager.hpp"

namespace gs {
    namespace gui {

        // Scene panel that integrates with existing GUI
        class ScenePanel {
        public:
            ScenePanel(std::shared_ptr<const TrainerManager> trainer_manager);
            ~ScenePanel();

            void render(bool* p_open);
            void setOnDatasetLoad(std::function<void(const std::filesystem::path&)> callback);

        private:
            // Panel state
            float m_panelWidth = 300.0f;

            // Image list data for dataset mode
            std::vector<std::filesystem::path> m_imagePaths;
            typedef int CamId;
            // cam path to cam id
            std::map<std::filesystem::path, CamId> m_PathToCamId;
            int m_selectedImageIndex = -1;
            std::filesystem::path m_currentDatasetPath;
            bool m_needsScrollToSelection = false;

            // PLY scene graph data for PLY mode
            struct PLYNode {
                std::string name;
                bool visible = true;
                bool selected = false;
                size_t gaussian_count = 0;
            };
            std::vector<PLYNode> m_plyNodes;
            int m_selectedPLYIndex = -1;

            // Tab management
            enum class TabType {
                Images,
                PLYs
            };
            TabType m_activeTab = TabType::PLYs; // Default to PLYs tab (prioritize PLYs if available)

            // Current mode
            enum class DisplayMode {
                Empty,
                PLYSceneGraph,
                DatasetImages
            };
            DisplayMode m_currentMode = DisplayMode::Empty;

            // Callbacks
            std::function<void(const std::filesystem::path&)> m_onDatasetLoad;

            // Image preview integration
            std::unique_ptr<class ImagePreview> m_imagePreview;
            bool m_showImagePreview = false;

            // for loading training cameras in scene panel
            std::shared_ptr<const TrainerManager> m_trainer_manager;

            // Helper methods for tab management
            void updateModeFromTab();
            bool hasImages() const;
            bool hasPLYs() const;

            // Event handlers
            void setupEventHandlers();
            void handleSceneLoaded(const events::state::SceneLoaded& event);
            void handleSceneCleared();
            void handlePLYAdded(const events::state::PLYAdded& event);
            void handlePLYRemoved(const events::state::PLYRemoved& event);
            void handleGoToCamView(const events::cmd::GoToCamView& event);
            void loadImageCams(const std::filesystem::path& path);
            void onImageSelected(const std::filesystem::path& imagePath);
            void onImageDoubleClicked(size_t imageIndex);

            // PLY scene graph rendering
            void renderPLYSceneGraph();
            void renderImageList();
            void updatePLYNodes();
        };

    } // namespace gui
} // namespace gs