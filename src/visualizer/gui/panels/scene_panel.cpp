#include "gui/panels/scene_panel.hpp"
#include "gui/windows/image_preview.hpp"
#include <algorithm>
#include <filesystem>
#include <format>
#include <imgui.h>
#include <print>
#include <ranges>

namespace gs::gui {

    // ScenePanel Implementation
    ScenePanel::ScenePanel(std::shared_ptr<const TrainerManager> trainer_manager) : m_trainer_manager(trainer_manager) {
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

        events::state::PLYAdded::when([this](const auto& event) {
            handlePLYAdded(event);
        });

        events::state::PLYRemoved::when([this](const auto& event) {
            handlePLYRemoved(event);
        });
    }

    void ScenePanel::handleSceneLoaded(const events::state::SceneLoaded& event) {
        // Switch mode based on type
        if (event.type == events::state::SceneLoaded::Type::PLY) {
            m_currentMode = DisplayMode::PLYSceneGraph;
            // Clear any existing nodes
            m_plyNodes.clear();
            m_selectedPLYIndex = -1;
            // Don't try to update nodes here - wait for PLYAdded events
        } else if (event.type == events::state::SceneLoaded::Type::Dataset) {
            m_currentMode = DisplayMode::DatasetImages;
            // Clear PLY nodes when switching to dataset mode
            m_plyNodes.clear();
            m_selectedPLYIndex = -1;
            // Load the image list from the dataset
            if (!event.path.empty()) {
                loadImageCams(event.path);
            }
        }
    }

    void ScenePanel::handleSceneCleared() {
        // Clear all data
        m_imagePaths.clear();
        m_selectedImageIndex = -1;
        m_plyNodes.clear();
        m_selectedPLYIndex = -1;
        m_currentMode = DisplayMode::Empty;
    }

    void ScenePanel::handlePLYAdded(const events::state::PLYAdded& event) {
        // Add or update the PLY node
        auto it = std::find_if(m_plyNodes.begin(), m_plyNodes.end(),
                               [&event](const PLYNode& node) { return node.name == event.name; });

        if (it != m_plyNodes.end()) {
            // Update existing node
            it->gaussian_count = event.total_gaussians; // This is actually the individual model's count
        } else {
            // Add new node
            PLYNode node;
            node.name = event.name;
            node.visible = true;
            node.selected = false;
            node.gaussian_count = event.total_gaussians; // This is actually the individual model's count
            m_plyNodes.push_back(node);
        }

        // If we're not in PLY mode yet, switch to it
        if (m_currentMode != DisplayMode::PLYSceneGraph) {
            m_currentMode = DisplayMode::PLYSceneGraph;
        }
    }

    void ScenePanel::handlePLYRemoved(const events::state::PLYRemoved& event) {
        // Remove the node from our list
        auto it = std::find_if(m_plyNodes.begin(), m_plyNodes.end(),
                               [&event](const PLYNode& node) { return node.name == event.name; });

        if (it != m_plyNodes.end()) {
            m_plyNodes.erase(it);

            // Reset selection if necessary
            if (m_selectedPLYIndex >= static_cast<int>(m_plyNodes.size())) {
                m_selectedPLYIndex = -1;
            }
        }
    }

    void ScenePanel::updatePLYNodes() {
        // Query scene for current nodes
        events::query::GetSceneInfo{}.emit();

        // For now, we'll rebuild the node list when we get events
        // In a more sophisticated implementation, we'd query the scene directly
    }

    void ScenePanel::render(bool* p_open) {
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.5f, 0.5f, 0.5f, 0.8f));

        if (!ImGui::Begin("Scene", p_open)) {
            ImGui::End();
            ImGui::PopStyleColor();
            return;
        }

        // Make buttons smaller to fit the narrow panel
        float button_width = ImGui::GetContentRegionAvail().x;

        // Common controls
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
            if (m_currentMode == DisplayMode::DatasetImages && !m_currentDatasetPath.empty()) {
                loadImageCams(m_currentDatasetPath);
            } else if (m_currentMode == DisplayMode::PLYSceneGraph) {
                updatePLYNodes();
            }
        }

        ImGui::SameLine();

        if (ImGui::Button("Clear", ImVec2(button_width * 0.48f, 0))) {
            // Clear everything
            handleSceneCleared();

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

        // Render appropriate view based on mode
        switch (m_currentMode) {
        case DisplayMode::PLYSceneGraph:
            renderPLYSceneGraph();
            break;
        case DisplayMode::DatasetImages:
            renderImageList();
            break;
        case DisplayMode::Empty:
        default:
            ImGui::Text("No data loaded.");
            ImGui::Text("Use 'Open File Browser' to load:");
            ImGui::BulletText("PLY file(s) for viewing");
            ImGui::BulletText("Dataset for training");
            break;
        }

        ImGui::End();
        ImGui::PopStyleColor();

        // Render image preview window if open
        if (m_showImagePreview && m_imagePreview) {
            m_imagePreview->render(&m_showImagePreview);
        }
    }

    void ScenePanel::renderPLYSceneGraph() {
        ImGui::Text("Scene Graph (PLY Mode)");
        ImGui::Separator();

        // Add PLY button
        if (ImGui::Button("Add PLY", ImVec2(-1, 0))) {
            // Open file browser for adding PLY
            events::cmd::ShowWindow{.window_name = "file_browser", .show = true}.emit();

            events::notify::Log{
                .level = events::notify::Log::Level::Info,
                .message = "Opening file browser to add PLY...",
                .source = "ScenePanel"}
                .emit();
        }

        ImGui::Separator();

        // Scene graph tree
        ImGui::BeginChild("SceneGraph", ImVec2(0, 0), true);

        if (!m_plyNodes.empty()) {
            ImGui::Text("Models (%zu):", m_plyNodes.size());
            ImGui::Separator();

            for (size_t i = 0; i < m_plyNodes.size(); ++i) {
                auto& node = m_plyNodes[i];

                // Create unique ID
                std::string node_id = std::format("{}##{}", node.name, i);

                // Node flags
                ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf |
                                           ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                           ImGuiTreeNodeFlags_SpanAvailWidth;
                if (node.selected) {
                    flags |= ImGuiTreeNodeFlags_Selected;
                }

                // Visibility checkbox
                ImGui::PushID(static_cast<int>(i));
                bool visible = node.visible;
                if (ImGui::Checkbox("##vis", &visible)) {
                    node.visible = visible;
                    events::cmd::SetPLYVisibility{
                        .name = node.name,
                        .visible = visible}
                        .emit();
                }
                ImGui::PopID();

                ImGui::SameLine();

                // Tree node
                ImGui::TreeNodeEx(node_id.c_str(), flags);

                // Show gaussian count
                ImGui::SameLine();
                ImGui::TextDisabled("(%zu)", node.gaussian_count);

                // Selection
                if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                    m_selectedPLYIndex = static_cast<int>(i);
                    // Update selection
                    for (auto& n : m_plyNodes) {
                        n.selected = false;
                    }
                    node.selected = true;

                    // Emit selection event
                    events::ui::NodeSelected{
                        .path = node.name,
                        .type = "PLY",
                        .metadata = {
                            {"name", node.name},
                            {"gaussians", std::to_string(node.gaussian_count)},
                            {"visible", node.visible ? "true" : "false"}}}
                        .emit();
                }

                // Right-click context menu
                if (ImGui::BeginPopupContextItem()) {
                    if (ImGui::MenuItem("Remove")) {
                        events::cmd::RemovePLY{.name = node.name}.emit();
                    }
                    ImGui::Separator();
                    bool menu_visible = node.visible;
                    if (ImGui::MenuItem("Visible", nullptr, &menu_visible)) {
                        node.visible = menu_visible;
                        events::cmd::SetPLYVisibility{
                            .name = node.name,
                            .visible = menu_visible}
                            .emit();
                    }
                    ImGui::EndPopup();
                }
            }

            // Show total gaussian count
            size_t total_gaussians = 0;
            for (const auto& node : m_plyNodes) {
                if (node.visible) {
                    total_gaussians += node.gaussian_count;
                }
            }
            ImGui::Separator();
            ImGui::Text("Total visible: %zu gaussians", total_gaussians);

        } else {
            ImGui::Text("No PLY models loaded.");
            ImGui::Text("Click 'Add PLY' to load models.");
        }

        ImGui::EndChild();
    }

    void ScenePanel::renderImageList() {
        // Image list view
        ImGui::BeginChild("ImageList", ImVec2(0, 0), true);

        if (!m_imagePaths.empty()) {
            ImGui::Text("Images (%zu):", m_imagePaths.size());
            ImGui::Separator();

            for (size_t i = 0; i < m_imagePaths.size(); ++i) {
                const auto& imagePath = m_imagePaths[i];
                std::string filename = imagePath.filename().string();

                // Create unique ID for ImGui by combining filename with index
                std::string unique_id = std::format("{}##{}", filename, i);

                // Check if this item is selected
                bool is_selected = (m_selectedImageIndex == static_cast<int>(i));

                if (ImGui::Selectable(unique_id.c_str(), is_selected)) {
                    m_selectedImageIndex = static_cast<int>(i);
                    onImageSelected(imagePath);
                }

                // Handle double-click to open image preview
                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    onImageDoubleClicked(i);
                }

                // Context menu for right-click - use unique ID
                std::string context_menu_id = std::format("context_menu_{}", i);
                if (ImGui::BeginPopupContextItem(context_menu_id.c_str())) {
                    if (ImGui::MenuItem("Go to Cam View")) {
                        // Get the camera data for this image
                        auto cam_data_it = m_PathToCamId.find(imagePath);
                        if (cam_data_it != m_PathToCamId.end()) {
                            // Emit the new GoToCamView command event with camera data
                            events::cmd::GoToCamView{
                                .cam_id = cam_data_it->second}
                                .emit();

                            // Log the action
                            events::notify::Log{
                                .level = events::notify::Log::Level::Info,
                                .message = std::format("Going to camera view for: {} (Camera ID: {})",
                                                       imagePath.filename().string(),
                                                       cam_data_it->second),
                                .source = "ScenePanel"}
                                .emit();
                        } else {
                            // Log warning if camera data not found
                            events::notify::Log{
                                .level = events::notify::Log::Level::Warning,
                                .message = std::format("Camera data not found for: {}", imagePath.filename().string()),
                                .source = "ScenePanel"}
                                .emit();
                        }
                    }
                    ImGui::EndPopup();
                }
            }
        } else {
            ImGui::Text("No images loaded.");
            ImGui::Text("Use 'Open File Browser' to load a dataset.");
        }

        ImGui::EndChild();
    }

    void ScenePanel::loadImageCams(const std::filesystem::path& path) {

        m_currentDatasetPath = path;
        m_imagePaths.clear();
        m_PathToCamId.clear();
        m_selectedImageIndex = -1;

        if (!m_trainer_manager) {
            std::cerr << "m_trainer_manager was not set" << std::endl;
            return;
        }

        auto cams = m_trainer_manager->getCamList();

        for (const auto& cam : cams) {
            m_imagePaths.emplace_back(cam->image_path());
            m_PathToCamId[cam->image_path()] = cam->uid();
        }

        // Sort paths for consistent ordering
        std::ranges::sort(m_imagePaths, [](const auto& a, const auto& b) {
            return a.filename() < b.filename();
        });

        // Log the action
        events::notify::Log{
            .level = events::notify::Log::Level::Info,
            .message = std::format("Loaded {} images from dataset: {}", m_imagePaths.size(), path.string()),
            .source = "ScenePanel"}
            .emit();
    }

    void ScenePanel::setOnDatasetLoad(std::function<void(const std::filesystem::path&)> callback) {
        m_onDatasetLoad = std::move(callback);
    }

    void ScenePanel::onImageSelected(const std::filesystem::path& imagePath) {
        // Log selection
        events::notify::Log{
            .level = events::notify::Log::Level::Debug,
            .message = std::format("Selected image: {}", imagePath.filename().string()),
            .source = "ScenePanel"}
            .emit();

        // Publish NodeSelectedEvent for other components to react
        events::ui::NodeSelected{
            .path = imagePath.string(),
            .type = "Images",
            .metadata = {{"filename", imagePath.filename().string()}, {"path", imagePath.string()}}}
            .emit();
    }

    void ScenePanel::onImageDoubleClicked(size_t imageIndex) {
        if (imageIndex >= m_imagePaths.size()) {
            return;
        }

        const auto& imagePath = m_imagePaths[imageIndex];

        // Open the image preview with all images and current index
        if (m_imagePreview) {
            m_imagePreview->open(m_imagePaths, imageIndex);
            m_showImagePreview = true;
        }

        // Log the action
        events::notify::Log{
            .level = events::notify::Log::Level::Info,
            .message = std::format("Opening image preview: {}", imagePath.filename().string()),
            .source = "ScenePanel"}
            .emit();
    }

} // namespace gs::gui