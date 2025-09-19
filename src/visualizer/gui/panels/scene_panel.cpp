/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/scene_panel.hpp"
#include "core/logger.hpp"
#include "gui/utils/native_dialogs.hpp"
#include "gui/windows/image_preview.hpp"
#include <algorithm>
#include <filesystem>
#include <format>
#include <imgui.h>
#include <ranges>
#include <stdexcept>

namespace gs::gui {

    // ScenePanel Implementation
    ScenePanel::ScenePanel(std::shared_ptr<const TrainerManager> trainer_manager) : m_trainer_manager(trainer_manager) {
        // Create image preview window
        m_imagePreview = std::make_unique<ImagePreview>();
        setupEventHandlers();
        LOG_DEBUG("ScenePanel created");
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

        // Listen for PLY visibility changes to update checkboxes
        events::cmd::SetPLYVisibility::when([this](const auto& event) {
            // Update the visibility state in our local PLY nodes
            auto it = std::find_if(m_plyNodes.begin(), m_plyNodes.end(),
                                   [&event](const PLYNode& node) { return node.name == event.name; });
            if (it != m_plyNodes.end()) {
                it->visible = event.visible;
                LOG_TRACE("Updated PLY '{}' visibility in scene panel to: {}", event.name, event.visible);
            }
        });

        // Listen for GoToCamView to sync selection
        events::cmd::GoToCamView::when([this](const auto& event) {
            handleGoToCamView(event);
        });
    }

    void ScenePanel::handleSceneLoaded(const events::state::SceneLoaded& event) {
        LOG_DEBUG("Scene loaded event - type: {}",
                  event.type == events::state::SceneLoaded::Type::PLY ? "PLY" : "Dataset");

        if (event.type == events::state::SceneLoaded::Type::PLY) {
            m_currentMode = DisplayMode::PLYSceneGraph;
            m_plyNodes.clear();
            m_selectedPLYIndex = -1;
            m_activeTab = TabType::PLYs; // Switch to PLY tab
            LOG_TRACE("Switched to PLY scene graph mode");
        } else if (event.type == events::state::SceneLoaded::Type::Dataset) {
            m_currentMode = DisplayMode::DatasetImages;
            m_plyNodes.clear();
            m_selectedPLYIndex = -1;
            m_activeTab = TabType::Images; // Switch to Images tab
            LOG_TRACE("Switched to dataset images mode");
            if (!event.path.empty()) {
                loadImageCams(event.path);
            }
        }
    }

    void ScenePanel::handleSceneCleared() {
        LOG_DEBUG("Clearing scene panel data");
        // Clear all data
        m_imagePaths.clear();
        m_selectedImageIndex = -1;
        m_plyNodes.clear();
        m_selectedPLYIndex = -1;
        m_currentMode = DisplayMode::Empty;
        // Keep the active tab as is - user might want to stay on the same tab
    }

    void ScenePanel::handlePLYAdded(const events::state::PLYAdded& event) {
        LOG_DEBUG("PLY added to scene panel: '{}' ({} gaussians, {} total)",
                  event.name, event.node_gaussians, event.total_gaussians);

        // Add or update the PLY node
        auto it = std::find_if(m_plyNodes.begin(), m_plyNodes.end(),
                               [&event](const PLYNode& node) { return node.name == event.name; });

        if (it != m_plyNodes.end()) {
            // Update existing node with its individual gaussian count
            it->gaussian_count = event.node_gaussians;
            LOG_TRACE("Updated existing PLY node '{}'", event.name);
        } else {
            // Add new node with its individual gaussian count
            PLYNode node;
            node.name = event.name;
            node.visible = event.is_visible;
            node.selected = false;
            node.gaussian_count = event.node_gaussians; // Use node-specific count
            m_plyNodes.push_back(node);
            LOG_TRACE("Added new PLY node '{}'", event.name);
        }

        // Update current mode based on active tab
        updateModeFromTab();
    }

    void ScenePanel::handlePLYRemoved(const events::state::PLYRemoved& event) {
        LOG_DEBUG("PLY removed from scene panel: '{}'", event.name);

        // Remove the node from our list
        auto it = std::find_if(m_plyNodes.begin(), m_plyNodes.end(),
                               [&event](const PLYNode& node) { return node.name == event.name; });

        if (it != m_plyNodes.end()) {
            m_plyNodes.erase(it);

            // Reset selection if necessary
            if (m_selectedPLYIndex >= static_cast<int>(m_plyNodes.size())) {
                m_selectedPLYIndex = -1;
                LOG_TRACE("Reset PLY selection index");
            }
        }

        // Update current mode based on active tab
        updateModeFromTab();
    }

    void ScenePanel::handleGoToCamView(const events::cmd::GoToCamView& event) {
        // Find the image path for this camera ID
        for (const auto& [path, cam_id] : m_PathToCamId) {
            if (cam_id == event.cam_id) {
                // Find index in sorted image list
                if (auto it = std::find(m_imagePaths.begin(), m_imagePaths.end(), path); it != m_imagePaths.end()) {
                    m_selectedImageIndex = static_cast<int>(std::distance(m_imagePaths.begin(), it));
                    m_needsScrollToSelection = true; // Mark that we need to scroll
                    LOG_TRACE("Synced image selection to camera ID {} (index {})",
                              event.cam_id, m_selectedImageIndex);
                }
                break;
            }
        }
    }

    void ScenePanel::updatePLYNodes() {
        LOG_TRACE("Updating PLY nodes");
        // For now, we'll rebuild the node list when we get events
        // In a more sophisticated implementation, we'd query the scene directly
    }

    void ScenePanel::updateModeFromTab() {
        // Update display mode based on active tab and available data
        // Prioritize PLYs if available and active tab is PLYs
        if (m_activeTab == TabType::PLYs && !m_plyNodes.empty()) {
            m_currentMode = DisplayMode::PLYSceneGraph;
            LOG_TRACE("Display mode set to PLYSceneGraph");
        } else if (m_activeTab == TabType::Images && !m_imagePaths.empty()) {
            m_currentMode = DisplayMode::DatasetImages;
            LOG_TRACE("Display mode set to DatasetImages");
        } else if (!m_plyNodes.empty()) {
            // Fall back to PLYs if available (even if Images tab was selected but no images)
            m_currentMode = DisplayMode::PLYSceneGraph;
            m_activeTab = TabType::PLYs;
            LOG_TRACE("Fallback to PLYSceneGraph mode");
        } else if (!m_imagePaths.empty()) {
            // Fall back to Images if PLYs not available
            m_currentMode = DisplayMode::DatasetImages;
            m_activeTab = TabType::Images;
            LOG_TRACE("Fallback to DatasetImages mode");
        } else {
            m_currentMode = DisplayMode::Empty;
            LOG_TRACE("Display mode set to Empty");
        }
    }

    bool ScenePanel::hasImages() const {
        return !m_imagePaths.empty();
    }

    bool ScenePanel::hasPLYs() const {
        return !m_plyNodes.empty();
    }

#ifdef WIN32
    void OpenProjectFileDialog() {
        // show native windows file dialog for project file selection
        PWSTR filePath = nullptr;

        COMDLG_FILTERSPEC rgSpec[] =
            {
                {L"LichtFeldStudio Project File", L"*.lfs;*.ls"},
            };

        if (SUCCEEDED(gs::gui::utils::selectFileNative(filePath, rgSpec, 1, false))) {
            std::filesystem::path project_path(filePath);
            events::cmd::LoadProject{.path = project_path}.emit();
            LOG_INFO("Loading project file : {}", std::filesystem::path(project_path).string());
        }
    }

    void OpenPlyFileDialog() {
        // show native windows file dialog for PLY file selection
        PWSTR filePath = nullptr;
        COMDLG_FILTERSPEC rgSpec[] =
            {
                {L"Point Cloud", L"*.ply;"},
            };
        if (SUCCEEDED(gs::gui::utils::selectFileNative(filePath, rgSpec, 1, false))) {
            std::filesystem::path ply_path(filePath);
            events::cmd::LoadFile{.path = ply_path}.emit();
            LOG_INFO("Loading PLY file : {}", std::filesystem::path(ply_path).string()); // FIXED: Changed from "Loading project file"
        }
    }

    void OpenDatasetFolderDialog() {
        // show native windows file dialog for folder selection
        PWSTR filePath = nullptr;
        if (SUCCEEDED(gs::gui::utils::selectFileNative(filePath, nullptr, 0, true))) {
            std::filesystem::path dataset_path(filePath);
            if (std::filesystem::is_directory(dataset_path)) {
                events::cmd::LoadFile{.path = dataset_path, .is_dataset = true}.emit();
                LOG_INFO("Loading dataset : {}", std::filesystem::path(dataset_path).string());
            }
        }
    }
#endif // WIN32

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
        if (ImGui::Button("Open Project", ImVec2(button_width, 0))) {
            // Fire the callback to open file browser with empty path
            if (m_onDatasetLoad) {
                m_onDatasetLoad(std::filesystem::path("")); // Empty path signals to open browser
            }
#ifdef WIN32
            // show native windows file dialog for project file selection
            OpenProjectFileDialog();

            // hide the file browser
            events::cmd::ShowWindow{.window_name = "file_browser", .show = false}.emit();
#endif // WIN32
        }

        if (ImGui::Button("Import dataset", ImVec2(button_width, 0))) {
            // Request to show file browser
            LOG_DEBUG("Opening file browser from scene panel");

            // Fire the callback to open file browser with empty path
            if (m_onDatasetLoad) {
                m_onDatasetLoad(std::filesystem::path("")); // Empty path signals to open browser
            }
#ifdef WIN32
            // show native windows file dialog for folder selection
            OpenDatasetFolderDialog();

            // hide the file browser
            events::cmd::ShowWindow{.window_name = "file_browser", .show = false}.emit();
#endif // WIN32
        }

        if (ImGui::Button("Open .ply", ImVec2(button_width, 0))) {
            // Request to show file browser
            LOG_DEBUG("Opening file browser from scene panel");

            // Fire the callback to open file browser with empty path
            if (m_onDatasetLoad) {
                m_onDatasetLoad(std::filesystem::path("")); // Empty path signals to open browser
            }
#ifdef WIN32
            // show native windows file dialog for folder selection
            OpenPlyFileDialog();

            // hide the file browser
            events::cmd::ShowWindow{.window_name = "file_browser", .show = false}.emit();
#endif // WIN32
        }

        if (ImGui::Button("Refresh", ImVec2(button_width * 0.48f, 0))) {
            if (m_currentMode == DisplayMode::DatasetImages && !m_currentDatasetPath.empty()) {
                LOG_DEBUG("Refreshing dataset images");
                loadImageCams(m_currentDatasetPath);
            } else if (m_currentMode == DisplayMode::PLYSceneGraph) {
                LOG_DEBUG("Refreshing PLY nodes");
                updatePLYNodes();
            }
        }

        ImGui::SameLine();

        if (ImGui::Button("Clear", ImVec2(button_width * 0.48f, 0))) {
            LOG_INFO("Clearing scene from panel");
            // Clear everything
            handleSceneCleared();

            // Also clear the actual scene data
            events::cmd::ClearScene{}.emit();
        }

        ImGui::Separator();

        // Render tabs if we have any data
        if (hasImages() || hasPLYs()) {
            if (ImGui::BeginTabBar("SceneTabs", ImGuiTabBarFlags_None)) {

                // PLYs tab - show first if we have PLYs (prioritize PLYs)
                if (hasPLYs()) {
                    bool plys_tab_selected = ImGui::BeginTabItem("PLYs");
                    if (plys_tab_selected) {
                        if (m_activeTab != TabType::PLYs) {
                            m_activeTab = TabType::PLYs;
                            m_currentMode = DisplayMode::PLYSceneGraph;
                            LOG_TRACE("Switched to PLYs tab");
                        }
                        renderPLYSceneGraph();
                        ImGui::EndTabItem();
                    }
                }

                // Images tab - show second
                if (hasImages()) {
                    bool images_tab_selected = ImGui::BeginTabItem("Images");
                    if (images_tab_selected) {
                        if (m_activeTab != TabType::Images) {
                            m_activeTab = TabType::Images;
                            m_currentMode = DisplayMode::DatasetImages;
                            LOG_TRACE("Switched to Images tab");
                        }
                        renderImageList();
                        ImGui::EndTabItem();
                    }
                }

                ImGui::EndTabBar();
            }
        } else {
            // No data loaded - show empty state
            ImGui::Text("No data loaded.");
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

        if (!m_plyNodes.empty() && m_plyNodes.size() > 1) {
            ImGui::TextDisabled("Tip: Press 'T' to cycle through PLYs");
        }

        ImGui::Separator();

        // Add PLY button
        if (ImGui::Button("Add PLY", ImVec2(-1, 0))) {
            // Open file browser for adding PLY
            events::cmd::ShowWindow{.window_name = "file_browser", .show = true}.emit();
            LOG_DEBUG("Opening file browser to add PLY");
#ifdef WIN32
            // show native windows file dialog for folder selection
            OpenPlyFileDialog();

            // hide the file browser
            events::cmd::ShowWindow{.window_name = "file_browser", .show = false}.emit();
#endif // WIN32
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
                std::string popup_id = std::format("popup_{}", i); // Unique popup ID

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
                    LOG_TRACE("PLY '{}' visibility changed to: {}", node.name, visible);
                }
                ImGui::PopID();

                ImGui::SameLine();

                // Tree node
                ImGui::TreeNodeEx(node_id.c_str(), flags);

                // Show gaussian count
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.2f, 0.2f, 0.2f, 1), "(%zu)", node.gaussian_count);

                // Selection
                if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                    m_selectedPLYIndex = static_cast<int>(i);
                    // Update selection
                    for (auto& n : m_plyNodes) {
                        n.selected = false;
                    }
                    node.selected = true;

                    LOG_DEBUG("PLY '{}' selected", node.name);

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

                // Right-click context menu - provide explicit popup ID
                if (ImGui::BeginPopupContextItem(popup_id.c_str())) {
                    if (ImGui::MenuItem("Remove")) {
                        events::cmd::RemovePLY{.name = node.name}.emit();
                        LOG_INFO("Removing PLY '{}' via context menu", node.name);
                    }
                    ImGui::Separator();
                    bool menu_visible = node.visible;
                    if (ImGui::MenuItem("Visible", nullptr, &menu_visible)) {
                        node.visible = menu_visible;
                        events::cmd::SetPLYVisibility{
                            .name = node.name,
                            .visible = menu_visible}
                            .emit();
                        LOG_TRACE("PLY '{}' visibility toggled via menu to: {}", node.name, menu_visible);
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

            // Track if we need to scroll to the selected item
            bool should_scroll = false;

            for (size_t i = 0; i < m_imagePaths.size(); ++i) {
                const auto& imagePath = m_imagePaths[i];
                std::string filename = imagePath.filename().string();

                // Create unique ID for ImGui by combining filename with index
                std::string unique_id = std::format("{}##{}", filename, i);

                // Check if this item is selected
                bool is_selected = (m_selectedImageIndex == static_cast<int>(i));

                // Push a different color for selected items to make them more visible
                if (is_selected) {
                    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.3f, 0.5f, 0.8f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.4f, 0.6f, 0.9f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.3f, 0.5f, 0.8f, 1.0f));
                }

                if (ImGui::Selectable(unique_id.c_str(), is_selected)) {
                    m_selectedImageIndex = static_cast<int>(i);
                    onImageSelected(imagePath);
                }

                // Scroll to this item if it's selected and we need to scroll
                if (is_selected && m_needsScrollToSelection) {
                    ImGui::SetScrollHereY(0.5f); // Center the selected item
                    m_needsScrollToSelection = false;
                    should_scroll = true;
                }

                if (is_selected) {
                    ImGui::PopStyleColor(3);
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

                            LOG_INFO("Going to camera view for: {} (Camera ID: {})",
                                     imagePath.filename().string(),
                                     cam_data_it->second);
                        } else {
                            // Log warning if camera data not found
                            LOG_WARN("Camera data not found for: {}", imagePath.filename().string());
                        }
                    }
                    ImGui::EndPopup();
                }
            }

            if (should_scroll) {
                LOG_TRACE("Scrolled to selected image at index {}", m_selectedImageIndex);
            }
        } else {
            ImGui::Text("No images loaded.");
            ImGui::Text("Use 'Open File Browser' to load a dataset.");
        }

        ImGui::EndChild();
    }

    void ScenePanel::loadImageCams(const std::filesystem::path& path) {
        LOG_TIMER_TRACE("ScenePanel::loadImageCams");

        m_currentDatasetPath = path;
        m_imagePaths.clear();
        m_PathToCamId.clear();
        m_selectedImageIndex = -1;

        if (!m_trainer_manager) {
            LOG_ERROR("m_trainer_manager was not set");
            return;
        }

        LOG_DEBUG("Loading camera list from dataset: {}", path.string());
        auto cams = m_trainer_manager->getCamList();
        LOG_DEBUG("Found {} cameras", cams.size());

        for (const auto& cam : cams) {
            m_imagePaths.emplace_back(cam->image_path());
            m_PathToCamId[cam->image_path()] = cam->uid();
            LOG_TRACE("Added camera: {} (ID: {})", cam->image_path().filename().string(), cam->uid());
        }

        // Sort paths for consistent ordering
        std::ranges::sort(m_imagePaths, [](const auto& a, const auto& b) {
            return a.filename() < b.filename();
        });

        LOG_INFO("Loaded {} images from dataset: {}", m_imagePaths.size(), path.string());
    }

    void ScenePanel::setOnDatasetLoad(std::function<void(const std::filesystem::path&)> callback) {
        m_onDatasetLoad = std::move(callback);
    }

    void ScenePanel::onImageSelected(const std::filesystem::path& imagePath) {
        LOG_DEBUG("Selected image: {}", imagePath.filename().string());

        // Publish NodeSelectedEvent for other components to react
        events::ui::NodeSelected{
            .path = imagePath.string(),
            .type = "Images",
            .metadata = {{"filename", imagePath.filename().string()}, {"path", imagePath.string()}}}
            .emit();
    }

    void ScenePanel::onImageDoubleClicked(size_t imageIndex) {
        if (imageIndex >= m_imagePaths.size()) {
            LOG_WARN("Invalid image index for double-click: {}", imageIndex);
            return;
        }

        const auto& imagePath = m_imagePaths[imageIndex];

        // Open the image preview with all images and current index
        if (m_imagePreview) {
            m_imagePreview->open(m_imagePaths, imageIndex);
            m_showImagePreview = true;
            LOG_INFO("Opening image preview: {} (index {})",
                     imagePath.filename().string(), imageIndex);
        }
    }

} // namespace gs::gui