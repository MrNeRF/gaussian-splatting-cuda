#pragma once

#include "visualizer/event_bus.hpp"
#include "visualizer/events.hpp"
#include <filesystem>
#include <functional>
#include <imgui.h>
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

            void Render();

            // Accessor for root node
            const SceneNode* GetRoot() const { return m_root.get(); }

        private:
            void RenderNode(SceneNode& node, int depth = 0);

            std::unique_ptr<SceneNode> m_root;
            SelectCallback m_onSelect;
            ExpandCallback m_onExpand;
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
            ScenePanel(EventBus& event_bus);
            ~ScenePanel();

            void render(bool* p_open);
            void loadColmapDataset(const std::filesystem::path& path);
            void setOnDatasetLoad(std::function<void(const std::filesystem::path&)> callback);

        private:
            EventBus& m_eventBus;
            TreeView m_treeView;
            ColmapDatasetLoader m_datasetLoader;

            // Panel state
            float m_panelWidth = 300.0f;

            // Callbacks
            std::function<void(const std::filesystem::path&)> m_onDatasetLoad;

            // Event handlers
            std::vector<size_t> m_eventHandlerIds;
            void setupEventHandlers();
            void handleSceneLoaded(const SceneLoadedEvent& event);
            void handleSceneCleared(const SceneClearedEvent& event);
            void onNodeSelected(const SceneNode& node);
            void onNodeExpanded(SceneNode& node);
        };

    } // namespace gui
} // namespace gs