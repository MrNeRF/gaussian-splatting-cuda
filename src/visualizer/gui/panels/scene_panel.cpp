#include "gui/panels/scene_panel.hpp"
#include "gui/windows/image_preview.hpp"
#include <algorithm>
#include <filesystem>
#include <format>
#include <imgui.h>
#include <print>
#include <ranges>

namespace gs::gui {

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

        // Check if this node represents a selected model
        bool is_selected = false;
        if (node.type == SceneNodeType::PointCloud && m_sceneRef) {
            for (const auto* entry : m_sceneRef->getModels()) {
                if (entry->path.string() == node.path && entry->selected) {
                    is_selected = true;
                    break;
                }
            }
        }

        if (is_selected) {
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

        // For point cloud models, add visibility checkbox
        if (node.type == SceneNodeType::PointCloud && m_sceneRef) {
            // Find the model entry
            const Scene::ModelEntry* model_entry = nullptr;
            for (const auto* entry : m_sceneRef->getModels()) {
                if (entry->path.string() == node.path) {
                    model_entry = entry;
                    break;
                }
            }

            if (model_entry) {
                bool visible = model_entry->visible;
                if (ImGui::Checkbox(std::format("##visible_{}", node.path).c_str(), &visible)) {
                    // Cast away const to modify visibility
                    const_cast<Scene*>(m_sceneRef)->setModelVisible(model_entry->id, visible);
                }
                ImGui::SameLine();
            }
        }

        std::string label = std::format("{} {}", icon, node.name);

        // Highlight selected models
        if (is_selected) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.2f, 0.8f, 0.2f, 1.0f));
        }

        bool isOpen = ImGui::TreeNodeEx(&node, flags, "%s", label.c_str());

        if (is_selected) {
            ImGui::PopStyleColor();
        }

        // Handle selection
        if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
            m_selectedNode = &node;

            // Handle model selection for point clouds
            if (node.type == SceneNodeType::PointCloud && m_sceneRef) {
                // Find the model ID from the path
                for (const auto* entry : m_sceneRef->getModels()) {
                    if (entry->path.string() == node.path) {
                        bool exclusive = !ImGui::GetIO().KeyCtrl; // Ctrl for multi-select
                        const_cast<Scene*>(m_sceneRef)->selectModel(entry->id, exclusive);
                        break;
                    }
                }
            }

            if (m_onSelect) {
                m_onSelect(node);
            }
        }

        // Handle double-click - check for double-click on the current item
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            if (m_onDoubleClick) {
                m_onDoubleClick(node);
            }
        }

        // Context menu
        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::MenuItem("Copy Path")) {
                ImGui::SetClipboardText(node.path.c_str());
            }

            // Only show "Load Dataset" option for dataset nodes that aren't the currently loaded dataset
            if (node.type == SceneNodeType::ColmapDataset) {
                // Check if this is already the loaded dataset by querying current state
                bool is_loaded = false;

                // Query current scene info
                events::query::SceneInfo response;
                auto handler = events::query::SceneInfo::when([&response](const auto& r) {
                    response = r;
                });
                events::query::GetSceneInfo{}.emit();

                // Check if this path matches the currently loaded dataset
                if (response.type == events::query::SceneInfo::Type::Dataset &&
                    response.source_path.string() == node.path) {
                    is_loaded = true;
                }

                if (!is_loaded && ImGui::MenuItem("Load Dataset")) {
                    // Emit the load file event directly from here
                    events::cmd::LoadFile{
                        .path = std::filesystem::path(node.path),
                        .is_dataset = true}
                        .emit();
                }
            }

            // Add remove option for selected models
            if (node.type == SceneNodeType::PointCloud && is_selected) {
                if (ImGui::MenuItem("Remove Selected")) {
                    events::cmd::RemoveSelectedModels{}.emit();
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
    ScenePanel::ScenePanel() {
        m_treeView.SetOnSelect([this](const SceneNode& node) {
            onNodeSelected(node);
        });

        m_treeView.SetOnExpand([this](SceneNode& node) {
            onNodeExpanded(node);
        });

        m_treeView.SetOnDoubleClick([this](const SceneNode& node) {
            onNodeDoubleClicked(node);
        });

        // Create image preview window
        m_imagePreview = std::make_unique<ImagePreview>();

        setupEventHandlers();
    }

    ScenePanel::~ScenePanel() {
        // Cleanup handled automatically
    }

    void ScenePanel::setupEventHandlers() {
        // Subscribe to events using the new event system
        events::state::SceneLoaded::when([this](const auto& event) {
            handleSceneLoaded(event);
        });

        events::state::SceneCleared::when([this](const auto&) {
            handleSceneCleared();
        });
    }

    void ScenePanel::handleSceneLoaded(const events::state::SceneLoaded& event) {
        // Set scene reference for the tree view
        m_treeView.SetSceneReference(event.scene);

        // Add the loaded scene to our tree
        if (!event.path.empty()) {
            if (event.type == events::state::SceneLoaded::Type::Dataset) {
                loadColmapDataset(event.path);
            } else if (event.type == events::state::SceneLoaded::Type::PLY) {
                // For PLY files, we need to append to existing tree or create new one
                const auto* currentRoot = m_treeView.GetRoot();
                std::unique_ptr<SceneNode> root;

                if (currentRoot && currentRoot->type == SceneNodeType::Root && currentRoot->name == "Loaded Models") {
                    // Clone existing tree (we need to recreate it because we can't modify const)
                    root = std::make_unique<SceneNode>();
                    root->name = "Loaded Models";
                    root->path = "";
                    root->type = SceneNodeType::Root;
                    root->expanded = true;

                    // Copy existing children
                    for (const auto& child : currentRoot->children) {
                        auto newChild = std::make_unique<SceneNode>();
                        newChild->name = child->name;
                        newChild->path = child->path;
                        newChild->type = child->type;
                        newChild->childrenLoaded = child->childrenLoaded;
                        newChild->metadata = child->metadata;
                        newChild->expanded = child->expanded;
                        root->children.push_back(std::move(newChild));
                    }
                } else {
                    // Create new root
                    root = std::make_unique<SceneNode>();
                    root->name = "Loaded Models";
                    root->path = "";
                    root->type = SceneNodeType::Root;
                    root->expanded = true;
                }

                // Add the new PLY file
                auto plyNode = std::make_unique<SceneNode>();
                plyNode->name = event.path.filename().string();
                plyNode->path = event.path.string();
                plyNode->type = SceneNodeType::PointCloud;
                plyNode->childrenLoaded = true;
                plyNode->metadata["gaussians"] = std::format("{}", event.num_gaussians);
                plyNode->metadata["format"] = "PLY";

                root->children.push_back(std::move(plyNode));
                root->childrenLoaded = true;

                m_treeView.SetRootNode(std::move(root));
            }
        }
    }

    void ScenePanel::handleSceneCleared() {
        // Clear the tree
        m_treeView.SetRootNode(nullptr);
        m_treeView.SetSceneReference(nullptr);
    }

    void ScenePanel::render(bool* p_open) {
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.5f, 0.5f, 0.5f, 0.8f));

        // Simplified flags - positioning is handled in GuiManager::render()
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove |
                                 ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoCollapse |
                                 ImGuiWindowFlags_NoTitleBar; // Add this for consistency

        if (!ImGui::Begin("Scene", p_open, flags)) {
            ImGui::End();
            ImGui::PopStyleColor();
            return;
        }

        // Add a custom title
        ImGui::Text("Scene");
        ImGui::Separator();

        // Make buttons smaller to fit the narrow panel
        float button_width = ImGui::GetContentRegionAvail().x;

        if (ImGui::Button("Open File Browser", ImVec2(button_width, 0))) {
            // Request to show file browser
            events::notify::Log{
                .level = events::notify::Log::Level::Info,
                .message = "Opening file browser...",
                .source = "ScenePanel"}
                .emit();

            // Fire the callback to open file browser with empty path
            if (m_onDatasetLoad) {
                m_onDatasetLoad(std::filesystem::path("")); // Empty path signals to open browser
            }
        }

        if (ImGui::Button("Refresh", ImVec2(button_width * 0.48f, 0))) {
            const auto* root = m_treeView.GetRoot();
            if (root) {
                auto path = root->path;
                loadColmapDataset(path);
            }
        }

        ImGui::SameLine();

        if (ImGui::Button("Clear", ImVec2(button_width * 0.48f, 0))) {
            // Clear the tree view
            m_treeView.SetRootNode(nullptr);

            // Also clear the actual scene data
            events::cmd::ClearScene{}.emit();

            // Log the action
            events::notify::Log{
                .level = events::notify::Log::Level::Info,
                .message = "Scene cleared",
                .source = "ScenePanel"}
                .emit();
        }

        ImGui::Separator();

        // Show total gaussian count if we have PLY models loaded
        const auto* root = m_treeView.GetRoot();
        if (root && root->type == SceneNodeType::Root && root->name == "Loaded Models") {
            size_t total_gaussians = 0;
            size_t visible_gaussians = 0;
            size_t selected_count = 0;

            // Get scene reference to check visibility
            auto* scene = m_treeView.GetSceneReference();

            for (const auto& child : root->children) {
                if (child->type == SceneNodeType::PointCloud) {
                    auto it = child->metadata.find("gaussians");
                    if (it != child->metadata.end()) {
                        try {
                            size_t count = std::stoull(it->second);
                            total_gaussians += count;

                            // Check if visible and selected
                            if (scene) {
                                for (const auto* entry : scene->getModels()) {
                                    if (entry->path.string() == child->path) {
                                        if (entry->visible) {
                                            visible_gaussians += count;
                                        }
                                        if (entry->selected) {
                                            selected_count++;
                                        }
                                        break;
                                    }
                                }
                            }
                        } catch (...) {}
                    }
                }
            }

            ImGui::Text("Total Gaussians: %zu", total_gaussians);
            ImGui::Text("Visible: %zu", visible_gaussians);
            ImGui::Text("Models: %zu (%zu selected)", root->children.size(), selected_count);
            ImGui::Separator();

            // Add help text
            ImGui::TextDisabled("Ctrl+Click to multi-select");
            ImGui::TextDisabled("Delete key to remove selected");
            ImGui::Separator();
        }

        // Tree view
        ImGui::BeginChild("SceneTree", ImVec2(0, 0), true);
        m_treeView.Render();
        ImGui::EndChild();

        ImGui::End();
        ImGui::PopStyleColor();

        // Render image preview window if open
        if (m_showImagePreview && m_imagePreview) {
            m_imagePreview->render(&m_showImagePreview);
        }
    }

    void ScenePanel::loadColmapDataset(const std::filesystem::path& path) {
        auto dataset = m_datasetLoader.LoadDataset(path);
        m_treeView.SetRootNode(std::move(dataset));

        // Log the action
        events::notify::Log{
            .level = events::notify::Log::Level::Info,
            .message = std::format("Loaded dataset: {}", path.string()),
            .source = "ScenePanel"}
            .emit();
    }

    void ScenePanel::setOnDatasetLoad(std::function<void(const std::filesystem::path&)> callback) {
        m_onDatasetLoad = std::move(callback);
    }

    void ScenePanel::onNodeSelected(const SceneNode& node) {
        // Log selection
        events::notify::Log{
            .level = events::notify::Log::Level::Debug,
            .message = std::format("Selected: {} ({})", node.name, node.path),
            .source = "ScenePanel"}
            .emit();

        // Publish NodeSelectedEvent for other components to react
        events::ui::NodeSelected{
            .path = node.path,
            .type = [&node]() -> std::string {
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
            .metadata = node.metadata}
            .emit();
    }

    void ScenePanel::onNodeExpanded(SceneNode& node) {
        if (node.type == SceneNodeType::Directory ||
            node.type == SceneNodeType::ColmapDataset ||
            node.type == SceneNodeType::Images ||
            node.type == SceneNodeType::Points3D) {
            m_datasetLoader.LoadDirectoryChildren(node);
        }
    }

    void ScenePanel::onNodeDoubleClicked(const SceneNode& node) {
        // Check if this is an image file
        if (node.type == SceneNodeType::Images && node.childrenLoaded) {

            // First, collect all image paths from the parent directory
            m_currentImagePaths.clear();

            // Find the parent node (should be the Images directory)
            const SceneNode* parent = nullptr;
            std::function<const SceneNode*(const SceneNode*, const SceneNode*)> findParent =
                [&findParent](const SceneNode* root, const SceneNode* target) -> const SceneNode* {
                for (const auto& child : root->children) {
                    if (child.get() == target) {
                        return root;
                    }
                    if (auto found = findParent(child.get(), target)) {
                        return found;
                    }
                }
                return nullptr;
            };

            if (m_treeView.GetRoot()) {
                parent = findParent(m_treeView.GetRoot(), &node);
            }

            // Collect all image paths from the parent
            if (parent) {
                collectImagePaths(*parent, m_currentImagePaths);
            } else {
                // Fallback: just use this single image
                m_currentImagePaths.push_back(node.path);
            }

            // Sort paths for consistent ordering
            std::ranges::sort(m_currentImagePaths);

            // Find the index of the current image
            std::filesystem::path node_path(node.path);
            auto it = std::ranges::find(m_currentImagePaths, node_path);
            size_t index = 0;
            if (it != m_currentImagePaths.end()) {
                index = std::distance(m_currentImagePaths.begin(), it);
            }

            // Open the image preview
            if (m_imagePreview) {
                m_imagePreview->open(m_currentImagePaths, index);
                m_showImagePreview = true;
            }

            // Log the action
            events::notify::Log{
                .level = events::notify::Log::Level::Info,
                .message = std::format("Opening image preview: {}", node.name),
                .source = "ScenePanel"}
                .emit();
        } else {
            std::println("[ScenePanel] Node is not an image file or is a directory");
        }
    }

    void ScenePanel::collectImagePaths(const SceneNode& node, std::vector<std::filesystem::path>& paths) const {
        // Check if this node is an image file
        if (node.type == SceneNodeType::Images && node.childrenLoaded) {
            paths.push_back(node.path);
        }

        // Recursively collect from children
        for (const auto& child : node.children) {
            collectImagePaths(*child, paths);
        }
    }

} // namespace gs::gui