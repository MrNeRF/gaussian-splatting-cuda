#pragma once

#include <condition_variable>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <torch/torch.h>
#include <vector>

// Existing functions
std::tuple<unsigned char*, int, int, int>
load_image(std::filesystem::path p, int res_div = -1);
void save_image(const std::filesystem::path& path, torch::Tensor image);
void save_image(const std::filesystem::path& path,
                const std::vector<torch::Tensor>& images,
                bool horizontal = true,
                int separator_width = 2);
void free_image(unsigned char* image);

// Batch image saving functionality
namespace image_io {

    class BatchImageSaver {
    public:
        // Singleton pattern to ensure cleanup on exit
        static BatchImageSaver& instance() {
            static BatchImageSaver instance;
            return instance;
        }

        // Delete copy/move constructors
        BatchImageSaver(const BatchImageSaver&) = delete;
        BatchImageSaver& operator=(const BatchImageSaver&) = delete;
        BatchImageSaver(BatchImageSaver&&) = delete;
        BatchImageSaver& operator=(BatchImageSaver&&) = delete;

        // Queue image for asynchronous saving
        void queue_save(const std::filesystem::path& path, torch::Tensor image);

        // Queue multiple images for side-by-side saving
        void queue_save_multiple(const std::filesystem::path& path,
                                 const std::vector<torch::Tensor>& images,
                                 bool horizontal = true,
                                 int separator_width = 2);

        // Wait for all pending saves to complete
        void wait_all();

        // Flush all pending saves and stop threads (called automatically on destruction)
        void shutdown();

        // Get number of pending saves
        size_t pending_count() const;

        // Enable/disable batch saving (useful for debugging)
        void set_enabled(bool enabled) { enabled_ = enabled; }
        bool is_enabled() const { return enabled_; }

    private:
        BatchImageSaver(size_t num_workers = 4);
        ~BatchImageSaver();

        struct SaveTask {
            std::filesystem::path path;
            torch::Tensor image;
            std::vector<torch::Tensor> images;
            bool is_multi;
            bool horizontal;
            int separator_width;
        };

        void worker_thread();
        void process_task(const SaveTask& task);

        std::vector<std::thread> workers_;
        std::queue<SaveTask> task_queue_;
        mutable std::mutex queue_mutex_;
        std::condition_variable cv_;
        std::condition_variable cv_finished_;
        std::atomic<bool> stop_{false};
        std::atomic<size_t> active_tasks_{0};
        std::atomic<bool> enabled_{true};
        size_t num_workers_;
    };

    // Convenience functions that use the singleton
    inline void save_image_async(const std::filesystem::path& path, torch::Tensor image) {
        BatchImageSaver::instance().queue_save(path, image);
    }

    inline void save_images_async(const std::filesystem::path& path,
                                  const std::vector<torch::Tensor>& images,
                                  bool horizontal = true,
                                  int separator_width = 2) {
        BatchImageSaver::instance().queue_save_multiple(path, images, horizontal, separator_width);
    }

    inline void wait_for_pending_saves() {
        BatchImageSaver::instance().wait_all();
    }

} // namespace image_io
