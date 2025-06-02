#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "core/image_io.hpp"
#include "external/stb_image.h"
#include "external/stb_image_resize.h"
#include "external/stb_image_write.h"

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// Thread pool for async image loading
class ImageLoadingPool {
private:
    struct Task {
        std::filesystem::path path;
        int res_div;
        std::promise<std::tuple<unsigned char*, int, int, int>> promise;
    };

    std::vector<std::thread> workers_;
    std::queue<Task> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};

    void worker() {
        while (!stop_) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                cv_.wait(lock, [this] { return !tasks_.empty() || stop_; });

                if (stop_)
                    break;

                task = std::move(tasks_.front());
                tasks_.pop();
            }

            try {
                auto result = load_image(task.path, task.res_div);
                task.promise.set_value(result);
            } catch (...) {
                task.promise.set_exception(std::current_exception());
            }
        }
    }

    ImageLoadingPool(int num_threads = 8) {
        for (int i = 0; i < num_threads; ++i) {
            workers_.emplace_back(&ImageLoadingPool::worker, this);
        }
    }

public:
    static ImageLoadingPool& instance() {
        static ImageLoadingPool pool;
        return pool;
    }

    ~ImageLoadingPool() {
        stop_ = true;
        cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    std::future<std::tuple<unsigned char*, int, int, int>>
    submit(const std::filesystem::path& path, int res_div) {
        Task task{path, res_div};
        auto future = task.promise.get_future();

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            tasks_.push(std::move(task));
        }
        cv_.notify_one();

        return future;
    }
};

// Synchronous image loading (existing implementation)
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
    std::cout << "Saving image: " << path << " shape: [" << height << ", " << width << ", " << channels << "]"
              << " min: " << image.min().item<float>()
              << " max: " << image.max().item<float>() << std::endl;

    // Convert to uint8
    auto img_uint8 = (image.clamp(0, 1) * 255).to(torch::kUInt8).contiguous();

    auto ext = path.extension().string();
    bool success = false;

    if (ext == ".png") {
        success = stbi_write_png(path.c_str(), width, height, channels,
                                 img_uint8.data_ptr<uint8_t>(), width * channels);
    } else if (ext == ".jpg" || ext == ".jpeg") {
        success = stbi_write_jpg(path.c_str(), width, height, channels,
                                 img_uint8.data_ptr<uint8_t>(), 95);
    }

    if (!success) {
        throw std::runtime_error("Failed to save image: " + path.string());
    }
}

void free_image(unsigned char* img) {
    stbi_image_free(img);
}

// Async image loading
std::future<std::tuple<unsigned char*, int, int, int>>
load_image_async(const std::filesystem::path& p, int res_div) {
    return ImageLoadingPool::instance().submit(p, res_div);
}