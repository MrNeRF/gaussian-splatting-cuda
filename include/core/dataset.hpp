#pragma once
#include "core/camera_info.hpp"
#include "core/camera.hpp"
#include "core/scene_info.hpp"
#include <vector>
#include <random>
#include <memory>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <atomic>

class DataLoader {
public:
    struct Config {
        bool shuffle;
        int num_workers;           // Number of worker threads
        int prefetch_factor;       // How many batches to prefetch per worker
        bool pin_memory;           // Pin memory for faster GPU transfer
        bool persistent_workers;   // Keep workers alive between epochs

        Config() : shuffle(true), num_workers(4), prefetch_factor(2),
                   pin_memory(true), persistent_workers(true) {}
    };

    DataLoader(std::unique_ptr<SceneInfo> scene_info,
               const gs::param::ModelParameters& params,
               const Config& config = Config());

    ~DataLoader();

    // Main interface
    Camera get_next_camera();

    // Iterator support for range-based for loops
    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = Camera;
        using difference_type = std::ptrdiff_t;
        using pointer = Camera*;
        using reference = Camera&;

        Iterator(DataLoader* loader, bool is_end = false)
            : _loader(loader), _is_end(is_end) {}

        Camera operator*() {
            if (_is_end) {
                throw std::runtime_error("Dereferencing end iterator");
            }
            return _loader->get_next_camera();
        }

        Iterator& operator++() {
            if (!_loader->has_more_data()) {
                _is_end = true;
            }
            return *this;
        }

        bool operator!=(const Iterator& other) const {
            return _is_end != other._is_end;
        }

    private:
        DataLoader* _loader;
        bool _is_end;
    };

    Iterator begin() { return Iterator(this, false); }
    Iterator end() { return Iterator(this, true); }

    // Status and control
    bool has_more_data() const;
    void reset(); // New epoch
    size_t size() const { return _camera_infos.size(); }
    int get_current_epoch() const { return _current_epoch; }
    const SceneInfo& get_scene_info() const { return *_scene_info; }

private:
    struct WorkItem {
        int camera_idx;
        int epoch;
        CameraInfo cam_info;
    };

    struct LoadedCamera {
        Camera camera;
        int epoch;
        int original_idx;

        LoadedCamera(Camera&& cam, int ep, int idx)
            : camera(std::move(cam)), epoch(ep), original_idx(idx) {}
    };

    // Configuration
    std::unique_ptr<SceneInfo> _scene_info;
    std::vector<CameraInfo> _camera_infos;
    const gs::param::ModelParameters& _params;
    Config _config;

    // State management
    std::vector<int> _indices;
    std::mt19937 _rng;
    std::atomic<size_t> _current_idx{0};
    std::atomic<int> _current_epoch{0};
    std::atomic<bool> _shutdown{false};

    // Worker threads and synchronization
    std::vector<std::thread> _workers;
    std::queue<WorkItem> _work_queue;
    std::queue<LoadedCamera> _ready_queue;

    mutable std::mutex _work_mutex;
    mutable std::mutex _ready_mutex;
    std::condition_variable _work_cv;
    std::condition_variable _ready_cv;

    // Methods
    void start_workers();
    void stop_workers();
    void worker_loop(int worker_id);
    void fill_work_queue();
    torch::Tensor load_image(const CameraInfo& cam_info);
    void prepare_next_epoch();
};

std::unique_ptr<DataLoader> create_dataloader(const gs::param::ModelParameters& params,
                                              int num_workers = 4,
                                              bool pin_memory = true);