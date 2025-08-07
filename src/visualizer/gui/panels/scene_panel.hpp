#pragma once

#include "core/events.hpp"
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gs {
    namespace gui {

        // Scene panel that integrates with existing GUI
        class ScenePanel {
        public:
            ScenePanel();
            ~ScenePanel();

            void render(bool* p_open);
            void setOnDatasetLoad(std::function<void(const std::filesystem::path&)> callback);

        private:
            // Panel state
            float m_panelWidth = 300.0f;

            // Image list data
            std::vector<std::filesystem::path> m_imagePaths;
            int m_selectedImageIndex = -1;
            std::filesystem::path m_currentDatasetPath;

            // Callbacks
            std::function<void(const std::filesystem::path&)> m_onDatasetLoad;

            // Image preview integration
            std::unique_ptr<class ImagePreview> m_imagePreview;
            bool m_showImagePreview = false;

            // Event handlers
            void setupEventHandlers();
            void handleSceneLoaded(const events::state::SceneLoaded& event);
            void handleSceneCleared();
            void loadImageList(const std::filesystem::path& path);
            void onImageSelected(const std::filesystem::path& imagePath);
            void onImageDoubleClicked(size_t imageIndex);
        };

    } // namespace gui
} // namespace gs