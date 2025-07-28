#include "visualizer/scene_panel.hpp"
#include <algorithm>
#include <format>
#include <imgui.h>
#include <iostream>
#include <print>
#include <ranges>

namespace gs {
    namespace gui {

        // TreeView Implementation
        void TreeView::Render() {
            if (!m_root)
                return;

            ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, 16.0f);
            RenderNode(*m_root);
            ImGui::PopStyleVar();
        }

        void TreeView::RenderNode(SceneNode& node, int depth) {
            ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_OpenOnArrow;

            if (node.children.empty() && node.childrenLoaded) {
                flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
            }

            if (m_selectedNode == &node) {
                flags |= ImGuiTreeNodeFlags_Selected;
            }

            // Icon based on node type - using simple text instead of unicode
            const char* icon = nullptr;
            switch (node.type) {
            case SceneNodeType::ColmapDataset: icon = "[D]"; break;
            case SceneNodeType::Directory: icon = "[+]"; break;
            case SceneNodeType::Images: icon = "[I]"; break;
            case SceneNodeType::Cameras: icon = "[C]"; break;
            case SceneNodeType::Points3D: icon = "[P]"; break;
            case SceneNodeType::TrainedModel: icon = "[M]"; break;
            case SceneNodeType::PointCloud: icon = "[PC]"; break;
            case SceneNodeType::File: icon = "[-]"; break;
            default: icon = "[?]";
            }

            std::string label = std::format("{} {}", icon, node.name);

            bool isOpen = ImGui::TreeNodeEx(&node, flags, "%s", label.c_str());

            // Handle selection
            if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                m_selectedNode = &node;
                if (m_onSelect) {
                    m_onSelect(node);
                }
            }

            // Context menu
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Copy Path")) {
                    ImGui::SetClipboardText(node.path.c_str());
                }
                if (node.type == SceneNodeType::ColmapDataset && ImGui::MenuItem("Load Dataset")) {
                    if (m_onSelect) {
                        m_onSelect(node);
                    }
                }
                ImGui::EndPopup();
            }

            // Tooltip with metadata
            if (ImGui::IsItemHovered() && !node.metadata.empty()) {
                ImGui::BeginTooltip();
                for (const auto& [key, value] : node.metadata) {
                    ImGui::Text("%s: %s", key.c_str(), value.c_str());
                }
                ImGui::EndTooltip();
            }

            if (isOpen && !(flags & ImGuiTreeNodeFlags_NoTreePushOnOpen)) {
                if (!node.childrenLoaded && m_onExpand) {
                    m_onExpand(node);
                }

                for (auto& child : node.children) {
                    RenderNode(*child, depth + 1);
                }

                ImGui::TreePop();
            }
        }

        // ColmapDatasetLoader Implementation
        std::unique_ptr<SceneNode> ColmapDatasetLoader::LoadDataset(const std::filesystem::path& path) {
            auto root = std::make_unique<SceneNode>();
            root->name = path.filename().string();
            root->path = path.string();
            root->type = SceneNodeType::ColmapDataset;

            root->metadata["dataset_type"] = "COLMAP";
            root->metadata["path"] = path.string();

            // Check if it's a valid COLMAP dataset
            bool hasImages = std::filesystem::exists(path / "images") || std::filesystem::exists(path / "Images");
            bool hasSparse = std::filesystem::exists(path / "sparse");

            if (hasImages || hasSparse) {
                root->metadata["valid"] = "true";
            }

            LoadDirectoryChildren(*root);

            return root;
        }

        void ColmapDatasetLoader::LoadDirectoryChildren(SceneNode& node) {
            if (node.childrenLoaded)
                return;

            std::filesystem::path dirPath(node.path);
            if (!std::filesystem::exists(dirPath) || !std::filesystem::is_directory(dirPath)) {
                node.childrenLoaded = true;
                return;
            }

            try {
                std::vector<std::filesystem::directory_entry> entries;

                // Use recursive_directory_iterator for image folders to get all images
                if (node.type == SceneNodeType::Images ||
                    (node.type == SceneNodeType::Directory && (node.name == "images" || node.name == "Images"))) {
                    // For image directories, get all image files recursively
                    for (const auto& entry : std::filesystem::recursive_directory_iterator(dirPath)) {
                        if (entry.is_regular_file()) {
                            std::string ext = entry.path().extension().string();
                            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" ||
                                ext == ".tiff" || ext == ".tif" || ext == ".webp") {
                                entries.push_back(entry);
                            }
                        }
                    }
                } else {
                    // For other directories, use regular iterator
                    for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
                        entries.push_back(entry);
                    }
                }

                // Sort: directories first, then by name
                std::ranges::sort(entries, [](const auto& a, const auto& b) {
                    if (a.is_directory() != b.is_directory()) {
                        return a.is_directory();
                    }
                    return a.path().filename() < b.path().filename();
                });

                for (const auto& entry : entries) {
                    auto child = std::make_unique<SceneNode>();

                    // For nested files in image directories, show relative path
                    if (node.type == SceneNodeType::Images) {
                        child->name = std::filesystem::relative(entry.path(), dirPath).string();
                    } else {
                        child->name = entry.path().filename().string();
                    }

                    child->path = entry.path().string();
                    child->type = DetermineNodeType(entry.path());

                    // Override type for files in image directories
                    if (node.type == SceneNodeType::Images && entry.is_regular_file()) {
                        child->type = SceneNodeType::Images;
                    }

                    if (entry.is_directory()) {
                        child->childrenLoaded = false;
                        // Special handling for known directories (case-insensitive)
                        std::string lowername = child->name;
                        std::transform(lowername.begin(), lowername.end(), lowername.begin(), ::tolower);
                        if (lowername == "images") {
                            child->type = SceneNodeType::Images;
                        } else if (lowername == "sparse" || lowername == "dense") {
                            child->type = SceneNodeType::Points3D;
                        }
                    } else {
                        child->childrenLoaded = true;
                        if (entry.is_regular_file()) {
                            auto size = entry.file_size();
                            if (size < 1024) {
                                child->metadata["size"] = std::format("{} B", size);
                            } else if (size < 1024 * 1024) {
                                child->metadata["size"] = std::format("{:.1f} KB", size / 1024.0);
                            } else {
                                child->metadata["size"] = std::format("{:.1f} MB", size / (1024.0 * 1024.0));
                            }
                            child->metadata["extension"] = entry.path().extension().string();
                        }
                    }

                    node.children.push_back(std::move(child));
                }
            } catch (const std::filesystem::filesystem_error& e) {
                // Handle permission errors, etc.
                std::println("Error loading directory: {}", e.what());
            }

            node.childrenLoaded = true;
        }

        SceneNodeType ColmapDatasetLoader::DetermineNodeType(const std::filesystem::path& path) {
            if (std::filesystem::is_directory(path)) {
                std::string dirname = path.filename().string();
                std::transform(dirname.begin(), dirname.end(), dirname.begin(), ::tolower);
                if (dirname == "images") {
                    return SceneNodeType::Images;
                } else if (dirname == "sparse" || dirname == "dense") {
                    return SceneNodeType::Points3D;
                }
                return SceneNodeType::Directory;
            }

            std::string ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            std::string filename = path.filename().string();
            std::string filename_lower = filename;
            std::transform(filename_lower.begin(), filename_lower.end(), filename_lower.begin(), ::tolower);

            // Camera files
            if ((ext == ".txt" || ext == ".bin") &&
                (filename_lower.find("cameras") != std::string::npos ||
                 filename_lower.find("intrinsics") != std::string::npos)) {
                return SceneNodeType::Cameras;
            }
            // Image files - comprehensive list
            else if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
                     ext == ".bmp" || ext == ".tiff" || ext == ".tif" ||
                     ext == ".webp" || ext == ".gif") {
                return SceneNodeType::Images;
            }
            // Point cloud files
            else if (ext == ".ply") {
                return SceneNodeType::PointCloud;
            }
            // 3D points files
            else if ((ext == ".bin" || ext == ".txt") &&
                     filename_lower.find("points3d") != std::string::npos) {
                return SceneNodeType::Points3D;
            }

            return SceneNodeType::File;
        }

        // ScenePanel Implementation
        ScenePanel::ScenePanel(EventBus& event_bus) : m_eventBus(event_bus) {
            m_treeView.SetOnSelect([this](const SceneNode& node) {
                onNodeSelected(node);
            });

            m_treeView.SetOnExpand([this](SceneNode& node) {
                onNodeExpanded(node);
            });

            setupEventHandlers();
        }

        ScenePanel::~ScenePanel() {
            // EventBus channels clean up automatically when EventBus is destroyed
            // No need to manually unsubscribe
        }

        void ScenePanel::setupEventHandlers() {
            // Subscribe to events using EventBus
            m_eventHandlerIds.push_back(
                m_eventBus.subscribe<SceneLoadedEvent>(
                    [this](const SceneLoadedEvent& event) {
                        handleSceneLoaded(event);
                    }));

            m_eventHandlerIds.push_back(
                m_eventBus.subscribe<SceneClearedEvent>(
                    [this](const SceneClearedEvent& event) {
                        handleSceneCleared(event);
                    }));
        }

        void ScenePanel::handleSceneLoaded(const SceneLoadedEvent& event) {
            // Add the loaded scene to our tree
            if (!event.source_path.empty()) {
                loadColmapDataset(event.source_path);
            }
        }

        void ScenePanel::handleSceneCleared(const SceneClearedEvent& event) {
            // Clear the tree
            m_treeView.SetRootNode(nullptr);
        }

        void ScenePanel::render(bool* p_open) {
            ImGui::SetNextWindowSize(ImVec2(m_panelWidth, 400), ImGuiCond_FirstUseEver);

            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.5f, 0.5f, 0.5f, 0.8f));

            if (!ImGui::Begin("Scene", p_open)) {
                ImGui::End();
                ImGui::PopStyleColor();
                return;
            }

            // Toolbar
            if (ImGui::Button("Open File Browser")) {
                // Request to show file browser
                m_eventBus.publish(LogMessageEvent{
                    LogMessageEvent::Level::Info,
                    "Opening file browser...",
                    std::string("ScenePanel")});

                // Fire the callback to open file browser with empty path
                if (m_onDatasetLoad) {
                    m_onDatasetLoad(std::filesystem::path("")); // Empty path signals to open browser
                }
            }

            ImGui::SameLine();
            if (ImGui::Button("Refresh")) {
                const auto* root = m_treeView.GetRoot();
                if (root) {
                    auto path = root->path;
                    loadColmapDataset(path);
                }
            }

            ImGui::SameLine();
            if (ImGui::Button("Clear")) {
                // Clear the tree view
                m_treeView.SetRootNode(nullptr);

                // Also clear the actual scene data
                m_eventBus.publish(ClearSceneCommand{});

                // Log the action
                m_eventBus.publish(LogMessageEvent{
                    LogMessageEvent::Level::Info,
                    "Scene cleared",
                    std::string("ScenePanel")});
            }

            ImGui::Separator();

            // Tree view
            ImGui::BeginChild("SceneTree", ImVec2(0, 0), true);
            m_treeView.Render();
            ImGui::EndChild();

            ImGui::End();
            ImGui::PopStyleColor();
        }

        void ScenePanel::loadColmapDataset(const std::filesystem::path& path) {
            auto dataset = m_datasetLoader.LoadDataset(path);
            m_treeView.SetRootNode(std::move(dataset));

            // Log the action
            m_eventBus.publish(LogMessageEvent{
                LogMessageEvent::Level::Info,
                std::format("Loaded dataset: {}", path.string()),
                std::string("ScenePanel")});
        }

        void ScenePanel::setOnDatasetLoad(std::function<void(const std::filesystem::path&)> callback) {
            m_onDatasetLoad = std::move(callback);
        }

        void ScenePanel::onNodeSelected(const SceneNode& node) {
            // Log selection
            m_eventBus.publish(LogMessageEvent{
                LogMessageEvent::Level::Debug,
                std::format("Selected: {} ({})", node.name, node.path),
                std::string("ScenePanel")});

            // If it's a dataset root and callback is set, trigger load
            if (node.type == SceneNodeType::ColmapDataset && m_onDatasetLoad) {
                m_onDatasetLoad(std::filesystem::path(node.path));
            }

            // Publish NodeSelectedEvent for other components to react
            m_eventBus.publish(NodeSelectedEvent{
                node.path,
                [&node]() -> std::string {
                    switch (node.type) {
                    case SceneNodeType::ColmapDataset: return "ColmapDataset";
                    case SceneNodeType::Directory: return "Directory";
                    case SceneNodeType::Images: return "Images";
                    case SceneNodeType::Cameras: return "Cameras";
                    case SceneNodeType::Points3D: return "Points3D";
                    case SceneNodeType::TrainedModel: return "TrainedModel";
                    case SceneNodeType::PointCloud: return "PointCloud";
                    case SceneNodeType::File: return "File";
                    default: return "Unknown";
                    }
                }(),
                node.metadata});
        }

        void ScenePanel::onNodeExpanded(SceneNode& node) {
            if (node.type == SceneNodeType::Directory ||
                node.type == SceneNodeType::ColmapDataset ||
                node.type == SceneNodeType::Images ||
                node.type == SceneNodeType::Points3D) {
                m_datasetLoader.LoadDirectoryChildren(node);
            }
        }

    } // namespace gui
} // namespace gs