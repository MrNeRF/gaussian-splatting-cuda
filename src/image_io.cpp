#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "core/image_io.hpp"
#include "external/stb_image.h"
#include "external/stb_image_resize.h"
#include "external/stb_image_write.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <vector>

// Existing implementations...
std::tuple<unsigned char*, int, int, int>
load_image(std::filesystem::path p, int res_div) {
    int w, h, c;
    unsigned char* img = stbi_load(p.string().c_str(), &w, &h, &c, 0);
    if (!img)
        throw std::runtime_error("Load failed: " + p.string() + " : " + stbi_failure_reason());

    if (res_div == 2 || res_div == 4 || res_div == 8) {
        int nw = w / res_div, nh = h / res_div;
        auto* out = static_cast<unsigned char*>(malloc(nw * nh * c));
        if (!stbir_resize_uint8(img, w, h, 0, out, nw, nh, 0, c))
            throw std::runtime_error("Resize failed: " + p.string() + " : " + stbi_failure_reason());
        stbi_image_free(img);
        img = out;
        w = nw;
        h = nh;
    }
    return {img, w, h, c};
}

void save_image(const std::filesystem::path& path, torch::Tensor image) {
    // Clone to avoid modifying original
    image = image.clone();

    // Ensure CPU and float
    image = image.to(torch::kCPU).to(torch::kFloat32);

    // Handle different input formats
    if (image.dim() == 4) { // [B, C, H, W] or [B, H, W, C]
        image = image.squeeze(0);
    }

    // Convert [C, H, W] to [H, W, C]
    if (image.dim() == 3 && image.size(0) <= 4) {
        image = image.permute({1, 2, 0});
    }

    // Make contiguous after permute
    image = image.contiguous();

    int height = image.size(0);
    int width = image.size(1);
    int channels = image.size(2);

    // Debug print
    std::cout << "Saving image: " << path << " shape: [" << height << ", " << width << ", " << channels << "]\n";

    // Convert to uint8
    auto img_uint8 = (image.clamp(0, 1) * 255).to(torch::kUInt8).contiguous();

    auto ext = path.extension().string();
    bool success = false;

    if (ext == ".png") {
        success = stbi_write_png(path.string().c_str(), width, height, channels,
                                 img_uint8.data_ptr<uint8_t>(), width * channels);
    } else if (ext == ".jpg" || ext == ".jpeg") {
        success = stbi_write_jpg(path.string().c_str(), width, height, channels,
                                 img_uint8.data_ptr<uint8_t>(), 95);
    }

    if (!success) {
        throw std::runtime_error("Failed to save image: " + path.string());
    }
}

void save_image(const std::filesystem::path& path,
                const std::vector<torch::Tensor>& images,
                bool horizontal,
                int separator_width) {
    if (images.empty()) {
        throw std::runtime_error("No images provided");
    }

    if (images.size() == 1) {
        save_image(path, images[0]);
        return;
    }

    // Prepare all images to same format
    std::vector<torch::Tensor> processed_images;
    for (auto img : images) {
        // Clone to avoid modifying original
        img = img.clone().to(torch::kCPU).to(torch::kFloat32);

        // Handle different input formats
        if (img.dim() == 4) {
            img = img.squeeze(0);
        }

        // Convert [C, H, W] to [H, W, C]
        if (img.dim() == 3 && img.size(0) <= 4) {
            img = img.permute({1, 2, 0});
        }

        processed_images.push_back(img.contiguous());
    }

    // Create separator (white by default)
    torch::Tensor separator;
    if (separator_width > 0) {
        auto first_img = processed_images[0];
        if (horizontal) {
            separator = torch::ones({first_img.size(0), separator_width, first_img.size(2)},
                                    first_img.options());
        } else {
            separator = torch::ones({separator_width, first_img.size(1), first_img.size(2)},
                                    first_img.options());
        }
    }

    // Concatenate images with separators
    torch::Tensor combined;
    for (size_t i = 0; i < processed_images.size(); ++i) {
        if (i == 0) {
            combined = processed_images[i];
        } else {
            if (separator_width > 0) {
                combined = torch::cat({combined, separator, processed_images[i]},
                                      horizontal ? 1 : 0);
            } else {
                combined = torch::cat({combined, processed_images[i]},
                                      horizontal ? 1 : 0);
            }
        }
    }

    // Save the combined image
    save_image(path, combined);
}

void free_image(unsigned char* img) {
    stbi_image_free(img);
}

// Batch image saver implementation
namespace image_io {

    BatchImageSaver::BatchImageSaver(size_t num_workers)
        : num_workers_(std::min(num_workers, std::min(size_t(8), size_t(std::thread::hardware_concurrency())))) {

        std::cout << "[BatchImageSaver] Starting with " << num_workers_ << " worker threads" << std::endl;

        for (size_t i = 0; i < num_workers_; ++i) {
            workers_.emplace_back(&BatchImageSaver::worker_thread, this);
        }
    }

    BatchImageSaver::~BatchImageSaver() {
        shutdown();
    }

    void BatchImageSaver::shutdown() {
        // Signal stop
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_)
                return; // Already stopped

            stop_ = true;
            std::cout << "[BatchImageSaver] Shutting down..." << std::endl;
        }
        cv_.notify_all();

        // Wait for all workers to finish
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }

        // Process any remaining tasks synchronously
        while (!task_queue_.empty()) {
            process_task(task_queue_.front());
            task_queue_.pop();
        }

        std::cout << "[BatchImageSaver] Shutdown complete" << std::endl;
    }

    void BatchImageSaver::queue_save(const std::filesystem::path& path, torch::Tensor image) {
        if (!enabled_) {
            save_image(path, image);
            return;
        }

        SaveTask task;
        task.path = path;
        task.image = image.clone(); // Clone to avoid data races
        task.is_multi = false;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                // If stopped, save synchronously
                save_image(path, image);
                return;
            }
            task_queue_.push(std::move(task));
            active_tasks_++;
        }
        cv_.notify_one();
    }

    void BatchImageSaver::queue_save_multiple(const std::filesystem::path& path,
                                              const std::vector<torch::Tensor>& images,
                                              bool horizontal,
                                              int separator_width) {
        if (!enabled_) {
            save_image(path, images, horizontal, separator_width);
            return;
        }

        SaveTask task;
        task.path = path;
        task.images.reserve(images.size());
        for (const auto& img : images) {
            task.images.push_back(img.clone());
        }
        task.is_multi = true;
        task.horizontal = horizontal;
        task.separator_width = separator_width;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                // If stopped, save synchronously
                save_image(path, images, horizontal, separator_width);
                return;
            }
            task_queue_.push(std::move(task));
            active_tasks_++;
        }
        cv_.notify_one();
    }

    void BatchImageSaver::wait_all() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        cv_finished_.wait(lock, [this] {
            return task_queue_.empty() && active_tasks_ == 0;
        });
    }

    size_t BatchImageSaver::pending_count() const {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        return task_queue_.size() + active_tasks_;
    }

    void BatchImageSaver::worker_thread() {
        while (true) {
            SaveTask task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                cv_.wait(lock, [this] { return stop_ || !task_queue_.empty(); });

                if (stop_ && task_queue_.empty()) {
                    break;
                }

                if (!task_queue_.empty()) {
                    task = std::move(task_queue_.front());
                    task_queue_.pop();
                } else {
                    continue;
                }
            }

            process_task(task);

            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                active_tasks_--;
            }
            cv_finished_.notify_all();
        }
    }

    void BatchImageSaver::process_task(const SaveTask& task) {
        try {
            if (task.is_multi) {
                save_image(task.path, task.images, task.horizontal, task.separator_width);
            } else {
                save_image(task.path, task.image);
            }
        } catch (const std::exception& e) {
            std::cerr << "[BatchImageSaver] Error saving " << task.path << ": " << e.what() << std::endl;
        }
    }

} // namespace image_io