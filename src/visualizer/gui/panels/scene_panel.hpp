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

        // Scene node types
        enum class SceneNodeType {
            Root,
            ColmapDataset,
            Directory,
            Images,
            Cameras,
            Points3D,
            File,
            TrainedModel,
            PointCloud
        };

        // Scene node data structure
        struct SceneNode {
            std::string name;
            std::string path;
            SceneNodeType type;
            std::vector<std::unique_ptr<SceneNode>> children;
            bool expanded = false;
            bool childrenLoaded = false;
            std::unordered_map<std::string, std::string> metadata;
        };

        // Tree view component
        class TreeView {
        public:
            using SelectCallback = std::function<void(const SceneNode&)>;
            using ExpandCallback = std::function<void(SceneNode&)>;

            TreeView() = default;

            void SetRootNode(std::unique_ptr<SceneNode> root) {
                m_root = std::move(root);
            }

            void SetOnSelect(SelectCallback callback) {
                m_onSelect = std::move(callback);
            }

            void SetOnExpand(ExpandCallback callback) {
                m_onExpand = std::move(callback);
            }

            void SetOnDoubleClick(SelectCallback callback) {
                m_onDoubleClick = std::move(callback);
            }

            void Render();

            // Accessor for root node
            const SceneNode* GetRoot() const { return m_root.get(); }

        private:
            void RenderNode(SceneNode& node, int depth = 0);

            std::unique_ptr<SceneNode> m_root;
            SelectCallback m_onSelect;
            ExpandCallback m_onExpand;
            SelectCallback m_onDoubleClick;
            SceneNode* m_selectedNode = nullptr;
        };

        // COLMAP dataset loader
        class ColmapDatasetLoader {
        public:
            std::unique_ptr<SceneNode> LoadDataset(const std::filesystem::path& path);
            void LoadDirectoryChildren(SceneNode& node);

        private:
            SceneNodeType DetermineNodeType(const std::filesystem::path& path);
        };

        // Scene panel that integrates with existing GUI
        class ScenePanel {
        public:
            ScenePanel();
            ~ScenePanel();

            void render(bool* p_open);
            void loadColmapDataset(const std::filesystem::path& path);
            void setOnDatasetLoad(std::function<void(const std::filesystem::path&)> callback);

        private:
            TreeView m_treeView;
            ColmapDatasetLoader m_datasetLoader;

            // Panel state
            float m_panelWidth = 300.0f;

            // Callbacks
            std::function<void(const std::filesystem::path&)> m_onDatasetLoad;

            // Image preview integration
            std::unique_ptr<class ImagePreview> m_imagePreview;
            std::vector<std::filesystem::path> m_currentImagePaths;
            bool m_showImagePreview = false;

            // Event handlers
            void setupEventHandlers();
            void handleSceneLoaded(const events::state::SceneLoaded& event);
            void handleSceneCleared();
            void onNodeSelected(const SceneNode& node);
            void onNodeExpanded(SceneNode& node);
            void onNodeDoubleClicked(const SceneNode& node);
            void collectImagePaths(const SceneNode& node, std::vector<std::filesystem::path>& paths) const;
        };

    } // namespace gui
} // namespace gs
