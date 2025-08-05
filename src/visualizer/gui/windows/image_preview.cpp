#include "gui/windows/image_preview.hpp"
#include "core/events.hpp"
#include "core/image_io.hpp"
#include <algorithm>
#include <format>
#include <future>
#include <glad/glad.h>
#include <imgui.h>
#include <print>
#include <thread>

namespace gs::gui {

    ImagePreview::ImagePreview() = default;

    ImagePreview::~ImagePreview() {
        close();
    }

    void ImagePreview::ensureMaxTextureSizeInitialized() {
        static std::once_flag initialized;
        std::call_once(initialized, [this]() {
            if (glGetIntegerv) { // Check if OpenGL is initialized
                glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_texture_size_);
            }
        });
    }

    void ImagePreview::open(const std::vector<std::filesystem::path>& image_paths, size_t initial_index) {

        if (image_paths.empty()) {
            events::notify::Warning{
                .message = "No images to preview"}
                .emit();
            return;
        }

        // Clear any existing state
        close();

        image_paths_ = image_paths;
        current_index_ = std::min(initial_index, image_paths.size() - 1);
        is_open_ = true;

        // Reset view
        zoom_ = 1.0f;
        pan_x_ = 0.0f;
        pan_y_ = 0.0f;
        fit_to_window_ = true;

        // Load current image
        if (!loadImage(image_paths_[current_index_])) {
            events::notify::Error{
                .message = "Failed to load image",
                .details = load_error_}
                .emit();
        }

        // Start preloading adjacent images
        preloadAdjacentImages();

        events::notify::Log{
            .level = events::notify::Log::Level::Info,
            .message = std::format("Opened image {}/{}: {}",
                                   current_index_ + 1,
                                   image_paths_.size(),
                                   image_paths_[current_index_].filename().string()),
            .source = "ImagePreview"}
            .emit();
    }

    void ImagePreview::open(const std::filesystem::path& image_path) {
        open(std::vector{image_path}, 0);
    }

    void ImagePreview::close() {
        is_open_ = false;
        image_paths_.clear();
        current_texture_.reset();
        prev_texture_.reset();
        next_texture_.reset();

        // Clear preload results
        {
            std::lock_guard<std::mutex> lock(preload_mutex_);
            prev_result_.reset();
            next_result_.reset();
        }

        load_error_.clear();
    }

    std::unique_ptr<ImageData> ImagePreview::loadImageData(const std::filesystem::path& path) {
        // Load image
        auto [data, width, height, channels] = ::load_image(path);

        // Wrap in RAII immediately
        auto image_data = std::make_unique<ImageData>(data, width, height, channels);

        // Validate
        if (!image_data->valid()) {
            throw std::runtime_error("Failed to load image data");
        }

        if (width <= 0 || height <= 0) {
            throw std::runtime_error(std::format("Invalid image dimensions: {}x{}", width, height));
        }

        if (channels < 1 || channels > 4) {
            throw std::runtime_error(std::format("Invalid number of channels: {}", channels));
        }

        return image_data;
    }

    std::unique_ptr<ImagePreview::ImageTexture> ImagePreview::createTexture(
        ImageData&& data, const std::filesystem::path& path) {

        ensureMaxTextureSizeInitialized();

        int width = data.width();
        int height = data.height();
        int channels = data.channels();

        // Check if we need to downscale
        if (width > max_texture_size_ || height > max_texture_size_) {
            // Calculate scale factor
            int scale_factor = 1;
            while (width / scale_factor > max_texture_size_ ||
                   height / scale_factor > max_texture_size_) {
                scale_factor *= 2;
            }

            // Reload at lower resolution
            auto scaled_data = loadImageData(path);
            if (!scaled_data) {
                throw std::runtime_error("Failed to reload image at lower resolution");
            }

            // Move the scaled data
            data = std::move(*scaled_data);
            width = data.width();
            height = data.height();
            channels = data.channels();
        }

        auto texture = std::make_unique<ImageTexture>();
        texture->width = width;
        texture->height = height;
        texture->path = path;

        // Clear any existing OpenGL errors
        while (glGetError() != GL_NO_ERROR) {}

        // Generate texture
        if (!texture->texture.generate()) {
            throw std::runtime_error("Failed to generate texture ID");
        }

        texture->texture.bind();

        // Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // Set pixel alignment
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        // Determine format
        GLenum format = GL_RGB;
        GLenum internal_format = GL_RGB8;

        switch (channels) {
        case 1:
            format = GL_RED;
            internal_format = GL_R8;
            break;
        case 3:
            format = GL_RGB;
            internal_format = GL_RGB8;
            break;
        case 4:
            format = GL_RGBA;
            internal_format = GL_RGBA8;
            break;
        default:
            throw std::runtime_error(std::format("Unsupported channel count: {}", channels));
        }

        // Upload texture
        glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0,
                     format, GL_UNSIGNED_BYTE, data.data());

        // Check for errors
        GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            std::string error_str;
            switch (error) {
            case GL_INVALID_ENUM: error_str = "GL_INVALID_ENUM"; break;
            case GL_INVALID_VALUE: error_str = "GL_INVALID_VALUE"; break;
            case GL_INVALID_OPERATION: error_str = "GL_INVALID_OPERATION"; break;
            case GL_OUT_OF_MEMORY: error_str = "GL_OUT_OF_MEMORY"; break;
            default: error_str = std::format("0x{:X}", error); break;
            }
            throw std::runtime_error(std::format("OpenGL error: {}", error_str));
        }

        // Set swizzle for grayscale
        if (channels == 1) {
            GLint swizzleMask[] = {GL_RED, GL_RED, GL_RED, GL_ONE};
            glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask);
        }

        GLTexture::unbind();

        return texture;
    }

    bool ImagePreview::loadImage(const std::filesystem::path& path) {
        try {
            is_loading_ = true;
            load_error_.clear();

            // Load image data with RAII
            auto image_data = loadImageData(path);

            // Create texture from data
            current_texture_ = createTexture(std::move(*image_data), path);

            is_loading_ = false;
            return true;

        } catch (const std::exception& e) {
            load_error_ = e.what();
            is_loading_ = false;
            std::println("[ImagePreview] Error loading image: {}", e.what());
            return false;
        }
    }

    void ImagePreview::preloadAdjacentImages() {
        // Don't start new preload if one is already in progress
        bool expected = false;
        if (!preload_in_progress_.compare_exchange_strong(expected, true)) {
            return;
        }

        // Clear existing preloaded data
        {
            std::lock_guard<std::mutex> lock(preload_mutex_);
            prev_result_.reset();
            next_result_.reset();
        }
        prev_texture_.reset();
        next_texture_.reset();

        // Ensure we have max texture size for the background threads
        ensureMaxTextureSizeInitialized();

        // Capture needed values
        GLint max_size = max_texture_size_;

        // Preload previous image
        if (current_index_ > 0) {
            std::thread([this, prev_idx = current_index_ - 1, max_size]() {
                try {
                    auto image_data = loadImageData(image_paths_[prev_idx]);

                    // Check if downscaling is needed
                    if (image_data->width() > max_size || image_data->height() > max_size) {
                        int scale = 2;
                        while (image_data->width() / scale > max_size ||
                               image_data->height() / scale > max_size) {
                            scale *= 2;
                        }

                        // Reload at lower resolution
                        auto [data, w, h, c] = ::load_image(image_paths_[prev_idx], scale);
                        image_data = std::make_unique<ImageData>(data, w, h, c);
                    }

                    auto result = std::make_unique<LoadResult>();
                    auto preloaded = std::make_unique<PreloadedImage>();
                    preloaded->data = std::move(*image_data);
                    preloaded->path = image_paths_[prev_idx];
                    result->image = std::move(preloaded);

                    std::lock_guard<std::mutex> lock(preload_mutex_);
                    prev_result_ = std::move(result);
                } catch (const std::exception& e) {
                    auto result = std::make_unique<LoadResult>();
                    result->error = e.what();

                    std::lock_guard<std::mutex> lock(preload_mutex_);
                    prev_result_ = std::move(result);
                }
            }).detach();
        }

        // Preload next image
        if (current_index_ + 1 < image_paths_.size()) {
            std::thread([this, next_idx = current_index_ + 1, max_size]() {
                try {
                    auto image_data = loadImageData(image_paths_[next_idx]);

                    // Check if downscaling is needed
                    if (image_data->width() > max_size || image_data->height() > max_size) {
                        int scale = 2;
                        while (image_data->width() / scale > max_size ||
                               image_data->height() / scale > max_size) {
                            scale *= 2;
                        }

                        // Reload at lower resolution
                        auto [data, w, h, c] = ::load_image(image_paths_[next_idx], scale);
                        image_data = std::make_unique<ImageData>(data, w, h, c);
                    }

                    auto result = std::make_unique<LoadResult>();
                    auto preloaded = std::make_unique<PreloadedImage>();
                    preloaded->data = std::move(*image_data);
                    preloaded->path = image_paths_[next_idx];
                    result->image = std::move(preloaded);

                    std::lock_guard<std::mutex> lock(preload_mutex_);
                    next_result_ = std::move(result);
                } catch (const std::exception& e) {
                    auto result = std::make_unique<LoadResult>();
                    result->error = e.what();

                    std::lock_guard<std::mutex> lock(preload_mutex_);
                    next_result_ = std::move(result);
                }

                preload_in_progress_ = false;
            }).detach();
        } else {
            preload_in_progress_ = false;
        }
    }

    void ImagePreview::checkPreloadedImages() {
        std::lock_guard<std::mutex> lock(preload_mutex_);

        // Check previous image
        if (prev_result_ && !prev_texture_) {
            if (prev_result_->image) {
                try {
                    prev_texture_ = createTexture(
                        std::move(prev_result_->image->data),
                        prev_result_->image->path);
                } catch (const std::exception& e) {
                    std::println("[ImagePreview] Failed to create prev texture: {}", e.what());
                }
            }
            prev_result_.reset();
        }

        // Check next image
        if (next_result_ && !next_texture_) {
            if (next_result_->image) {
                try {
                    next_texture_ = createTexture(
                        std::move(next_result_->image->data),
                        next_result_->image->path);
                } catch (const std::exception& e) {
                    std::println("[ImagePreview] Failed to create next texture: {}", e.what());
                }
            }
            next_result_.reset();
        }
    }

    void ImagePreview::nextImage() {
        if (image_paths_.empty() || current_index_ + 1 >= image_paths_.size()) {
            return;
        }

        current_index_++;

        // Always reset view when changing images
        zoom_ = 1.0f;
        pan_x_ = 0.0f;
        pan_y_ = 0.0f;

        // Use preloaded texture if available
        if (next_texture_) {
            current_texture_ = std::move(next_texture_);
        } else {
            if (!loadImage(image_paths_[current_index_])) {
                events::notify::Error{
                    .message = "Failed to load next image",
                    .details = load_error_}
                    .emit();
            }
        }

        preloadAdjacentImages();
    }

    void ImagePreview::previousImage() {
        if (image_paths_.empty() || current_index_ == 0) {
            return;
        }

        current_index_--;

        // Always reset view when changing images
        zoom_ = 1.0f;
        pan_x_ = 0.0f;
        pan_y_ = 0.0f;

        // Use preloaded texture if available
        if (prev_texture_) {
            current_texture_ = std::move(prev_texture_);
        } else {
            if (!loadImage(image_paths_[current_index_])) {
                events::notify::Error{
                    .message = "Failed to load previous image",
                    .details = load_error_}
                    .emit();
            }
        }

        preloadAdjacentImages();
    }

    void ImagePreview::goToImage(size_t index) {
        if (index >= image_paths_.size()) {
            return;
        }

        current_index_ = index;

        // Always reset view
        zoom_ = 1.0f;
        pan_x_ = 0.0f;
        pan_y_ = 0.0f;

        if (!loadImage(image_paths_[current_index_])) {
            events::notify::Error{
                .message = "Failed to load image",
                .details = load_error_}
                .emit();
        }

        preloadAdjacentImages();
    }

    std::pair<float, float> ImagePreview::calculateDisplaySize(int window_width, int window_height) const {
        if (!current_texture_) {
            return {0.0f, 0.0f};
        }

        float img_width = static_cast<float>(current_texture_->width);
        float img_height = static_cast<float>(current_texture_->height);

        if (fit_to_window_) {
            float scale_x = window_width / img_width;
            float scale_y = window_height / img_height;
            float scale = std::min(scale_x, scale_y) * 0.9f;

            return {img_width * scale * zoom_, img_height * scale * zoom_};
        } else {
            return {img_width * zoom_, img_height * zoom_};
        }
    }

    void ImagePreview::render(bool* p_open) {
        if (!is_open_ || !p_open || !*p_open) {
            close();
            return;
        }

        // Check for preloaded images and convert to textures
        checkPreloadedImages();

        // Window setup
        ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoScrollbar |
                                        ImGuiWindowFlags_NoScrollWithMouse |
                                        ImGuiWindowFlags_MenuBar;

        std::string title = "Image Preview";
        if (!image_paths_.empty()) {
            title = std::format("Image Preview - {}/{} - {}",
                                current_index_ + 1,
                                image_paths_.size(),
                                image_paths_[current_index_].filename().string());
        }

        if (!ImGui::Begin(title.c_str(), p_open, window_flags)) {
            ImGui::End();
            return;
        }

        // Menu bar
        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("View")) {
                ImGui::MenuItem("Fit to Window", "F", &fit_to_window_);
                ImGui::Separator();
                if (ImGui::MenuItem("Reset View", "R") || ImGui::MenuItem("Actual Size", "1")) {
                    zoom_ = 1.0f;
                    pan_x_ = 0.0f;
                    pan_y_ = 0.0f;
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Navigate")) {
                if (ImGui::MenuItem("Previous", "Left", nullptr, current_index_ > 0)) {
                    previousImage();
                }
                if (ImGui::MenuItem("Next", "Right", nullptr,
                                    current_index_ + 1 < image_paths_.size())) {
                    nextImage();
                }
                ImGui::Separator();
                if (ImGui::MenuItem("First", "Home", nullptr, !image_paths_.empty())) {
                    goToImage(0);
                }
                if (ImGui::MenuItem("Last", "End", nullptr, !image_paths_.empty())) {
                    goToImage(image_paths_.size() - 1);
                }
                ImGui::EndMenu();
            }

            ImGui::Separator();
            if (!image_paths_.empty()) {
                ImGui::Text("Image %zu/%zu", current_index_ + 1, image_paths_.size());
            }

            ImGui::EndMenuBar();
        }

        ImVec2 content_size = ImGui::GetContentRegionAvail();

        if (is_loading_) {
            ImGui::SetCursorPos(ImVec2(content_size.x * 0.5f - 50, content_size.y * 0.5f));
            ImGui::Text("Loading...");
            ImGui::End();
            return;
        }

        if (!load_error_.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Error: %s", load_error_.c_str());
            ImGui::End();
            return;
        }

        if (!current_texture_ || !current_texture_->texture.valid()) {
            ImGui::Text("No image loaded");
            ImGui::End();
            return;
        }

        auto [display_width, display_height] = calculateDisplaySize(
            static_cast<int>(content_size.x),
            static_cast<int>(content_size.y));

        float x_offset = (content_size.x - display_width) * 0.5f + pan_x_;
        float y_offset = (content_size.y - display_height) * 0.5f + pan_y_;

        ImGui::SetCursorPos(ImVec2(x_offset, y_offset + ImGui::GetCursorPosY()));
        ImGui::Image(
            (ImTextureID)(uintptr_t)current_texture_->texture.id(),
            ImVec2(display_width, display_height));

        if (ImGui::IsItemHovered()) {
            float wheel = ImGui::GetIO().MouseWheel;
            if (wheel != 0.0f) {
                float zoom_delta = wheel * 0.1f;
                zoom_ = std::clamp(zoom_ + zoom_delta, 0.1f, 10.0f);
            }

            if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
                ImVec2 delta = ImGui::GetIO().MouseDelta;
                pan_x_ += delta.x;
                pan_y_ += delta.y;
            }

            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                zoom_ = 1.0f;
                pan_x_ = 0.0f;
                pan_y_ = 0.0f;
            }
        }

        if (ImGui::IsWindowFocused()) {
            if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) {
                previousImage();
            }
            if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) {
                nextImage();
            }
            if (ImGui::IsKeyPressed(ImGuiKey_Home)) {
                goToImage(0);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_End) && !image_paths_.empty()) {
                goToImage(image_paths_.size() - 1);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_F)) {
                fit_to_window_ = !fit_to_window_;
            }
            if (ImGui::IsKeyPressed(ImGuiKey_R) || ImGui::IsKeyPressed(ImGuiKey_1)) {
                zoom_ = 1.0f;
                pan_x_ = 0.0f;
                pan_y_ = 0.0f;
            }

            if (ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || ImGui::IsKeyDown(ImGuiKey_RightCtrl)) {
                if (ImGui::IsKeyPressed(ImGuiKey_Equal) || ImGui::IsKeyPressed(ImGuiKey_KeypadAdd)) {
                    zoom_ = std::clamp(zoom_ + 0.1f, 0.1f, 10.0f);
                }
                if (ImGui::IsKeyPressed(ImGuiKey_Minus) || ImGui::IsKeyPressed(ImGuiKey_KeypadSubtract)) {
                    zoom_ = std::clamp(zoom_ - 0.1f, 0.1f, 10.0f);
                }
                if (ImGui::IsKeyPressed(ImGuiKey_0) || ImGui::IsKeyPressed(ImGuiKey_Keypad0)) {
                    zoom_ = 1.0f;
                }
            }
        }

        ImGui::End();
    }

} // namespace gs::gui
