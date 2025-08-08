#pragma once

#include "core/events.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <loader/loader.hpp>
#include "loader/formats/colmap.hpp"
#include "training/training_manager.hpp"

namespace gs {
    namespace gui {

        // Scene panel that integrates with existing GUI
        class ScenePanel {
        public:
            ScenePanel();
            ~ScenePanel();

            void render(bool* p_open);
            void setOnDatasetLoad(std::function<void(const std::filesystem::path&)> callback);

            void setTrainingManager( std::shared_ptr<const TrainerManager> trainer_manager) {m_trainer_manager=trainer_manager;}
        private:
            // Panel state
            float m_panelWidth = 300.0f;

            // Image list data
            std::vector<std::filesystem::path> m_imagePaths;
            typedef int CamId;
            // cam path to cam id
            std::map<std::filesystem::path, CamId> m_PathToCamId;

            int m_selectedImageIndex = -1;
            std::filesystem::path m_currentDatasetPath;

            // Callbacks
            std::function<void(const std::filesystem::path&)> m_onDatasetLoad;

            // Image preview integration
            std::unique_ptr<class ImagePreview> m_imagePreview;
            bool m_showImagePreview = false;

            // for loading training cameras in scene panel
            std::shared_ptr<const TrainerManager> m_trainer_manager;

            // Event handlers
            void setupEventHandlers();
            void handleSceneLoaded(const events::state::SceneLoaded& event);
            void handleSceneCleared();
            void loadImageCams(const std::filesystem::path& path);
            void onImageSelected(const std::filesystem::path& imagePath);
            void onImageDoubleClicked(size_t imageIndex);
        };

    } // namespace gui
} // namespace gs