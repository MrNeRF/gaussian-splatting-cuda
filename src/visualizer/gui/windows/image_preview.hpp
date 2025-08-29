/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/image_io.hpp"
#include <atomic>
#include <filesystem>
#include <glad/glad.h>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

// Forward declarations
struct GLFWwindow;

namespace gs::gui {

    /**
     * @brief RAII wrapper for raw image data
     */
    class ImageData {
    public:
        ImageData() = default;
        ImageData(unsigned char* data, int width, int height, int channels)
            : data_(data),
              width_(width),
              height_(height),
              channels_(channels) {}

        ~ImageData() {
            if (data_) {
                ::free_image(data_);
            }
        }

        // Delete copy operations
        ImageData(const ImageData&) = delete;
        ImageData& operator=(const ImageData&) = delete;

        // Move operations
        ImageData(ImageData&& other) noexcept
            : data_(std::exchange(other.data_, nullptr)),
              width_(std::exchange(other.width_, 0)),
              height_(std::exchange(other.height_, 0)),
              channels_(std::exchange(other.channels_, 0)) {}

        ImageData& operator=(ImageData&& other) noexcept {
            if (this != &other) {
                if (data_) {
                    ::free_image(data_);
                }
                data_ = std::exchange(other.data_, nullptr);
                width_ = std::exchange(other.width_, 0);
                height_ = std::exchange(other.height_, 0);
                channels_ = std::exchange(other.channels_, 0);
            }
            return *this;
        }

        // Accessors
        unsigned char* data() const { return data_; }
        int width() const { return width_; }
        int height() const { return height_; }
        int channels() const { return channels_; }
        bool valid() const { return data_ != nullptr; }

        // Release ownership
        unsigned char* release() {
            return std::exchange(data_, nullptr);
        }

    private:
        unsigned char* data_ = nullptr;
        int width_ = 0;
        int height_ = 0;
        int channels_ = 0;
    };

    /**
     * @brief RAII wrapper for OpenGL texture
     */
    class GLTexture {
    public:
        GLTexture() = default;

        ~GLTexture() {
            release();
        }

        // Delete copy operations
        GLTexture(const GLTexture&) = delete;
        GLTexture& operator=(const GLTexture&) = delete;

        // Move operations
        GLTexture(GLTexture&& other) noexcept
            : id_(std::exchange(other.id_, 0)) {}

        GLTexture& operator=(GLTexture&& other) noexcept {
            if (this != &other) {
                release();
                id_ = std::exchange(other.id_, 0);
            }
            return *this;
        }

        // Generate texture
        bool generate() {
            release();
            glGenTextures(1, &id_);
            return id_ != 0;
        }

        // Get ID
        GLuint id() const { return id_; }
        bool valid() const { return id_ != 0; }

        // Bind/unbind
        void bind() const {
            glBindTexture(GL_TEXTURE_2D, id_);
        }

        static void unbind() {
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        // Release
        void release() {
            if (id_ != 0) {
                glDeleteTextures(1, &id_);
                id_ = 0;
            }
        }

    private:
        GLuint id_ = 0;
    };

    /**
     * @brief Image preview window with navigation support
     */
    class ImagePreview {
    public:
        ImagePreview();
        ~ImagePreview();

        // Delete copy operations
        ImagePreview(const ImagePreview&) = delete;
        ImagePreview& operator=(const ImagePreview&) = delete;

        void open(const std::vector<std::filesystem::path>& image_paths, size_t initial_index);
        void open(const std::filesystem::path& image_path);
        void render(bool* p_open);
        bool isOpen() const { return is_open_; }
        void close();
        void nextImage();
        void previousImage();
        void goToImage(size_t index);

        std::optional<size_t> getCurrentIndex() const {
            return image_paths_.empty() ? std::nullopt : std::optional(current_index_);
        }

        size_t getImageCount() const { return image_paths_.size(); }

    private:
        // Image texture data
        struct ImageTexture {
            GLTexture texture;
            int width = 0;
            int height = 0;
            std::filesystem::path path;
        };

        // Preloaded image data
        struct PreloadedImage {
            ImageData data;
            std::filesystem::path path;
        };

        // Thread-safe loading result
        struct LoadResult {
            std::unique_ptr<PreloadedImage> image;
            std::string error;
        };

        // Helper methods
        void ensureMaxTextureSizeInitialized();
        std::unique_ptr<ImageData> loadImageData(const std::filesystem::path& path);
        std::unique_ptr<ImageTexture> createTexture(ImageData&& data, const std::filesystem::path& path);
        bool loadImage(const std::filesystem::path& path);
        void preloadAdjacentImages();
        void checkPreloadedImages();
        std::pair<float, float> calculateDisplaySize(int window_width, int window_height) const;

        // State
        bool is_open_ = false;
        std::vector<std::filesystem::path> image_paths_;
        size_t current_index_ = 0;

        // Current image texture
        std::unique_ptr<ImageTexture> current_texture_;

        // Thread-safe preload results
        mutable std::mutex preload_mutex_;
        std::unique_ptr<LoadResult> prev_result_;
        std::unique_ptr<LoadResult> next_result_;

        // Preloaded textures
        std::unique_ptr<ImageTexture> prev_texture_;
        std::unique_ptr<ImageTexture> next_texture_;

        // Loading state
        std::atomic<bool> is_loading_{false};
        std::string load_error_;
        std::atomic<bool> preload_in_progress_{false};

        // UI state
        float zoom_ = 1.0f;
        float pan_x_ = 0.0f;
        float pan_y_ = 0.0f;
        bool fit_to_window_ = true;

        // OpenGL limits
        GLint max_texture_size_ = 4096;
    };

} // namespace gs::gui
