#include "core/dataset.hpp"
#include "core/camera_utils.hpp"
#include "core/parameters.hpp"
#include "core/read_utils.hpp"
#include <algorithm>
#include <iostream>

DataLoader::DataLoader(std::unique_ptr<SceneInfo> scene_info,
                       const gs::param::ModelParameters& params,
                       const Config& config)
    : _scene_info(std::move(scene_info)),
      _camera_infos(std::move(_scene_info->_cameras)),
      _params(params),
      _config(config),
      _rng(std::random_device{}()) {

    // Initialize indices
    _indices.resize(_camera_infos.size());
    std::iota(_indices.begin(), _indices.end(), 0);

    if (_config.shuffle) {
        std::shuffle(_indices.begin(), _indices.end(), _rng);
    }

    std::cout << "DataLoader initialized with " << _camera_infos.size()
              << " cameras, " << _config.num_workers << " workers" << std::endl;

    // Start worker threads
    if (_config.num_workers > 0) {
        start_workers();
        fill_work_queue(); // Initial work queue fill
    }
}

DataLoader::~DataLoader() {
    stop_workers();
}

void DataLoader::start_workers() {
    _workers.reserve(_config.num_workers);
    for (int i = 0; i < _config.num_workers; ++i) {
        _workers.emplace_back(&DataLoader::worker_loop, this, i);
    }
}

void DataLoader::stop_workers() {
    _shutdown = true;
    _work_cv.notify_all();

    for (auto& worker : _workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    _workers.clear();
}

void DataLoader::worker_loop(int worker_id) {
    std::cout << "Worker " << worker_id << " started" << std::endl;

    while (!_shutdown) {
        WorkItem work_item;

        // Get work item
        {
            std::unique_lock<std::mutex> lock(_work_mutex);
            _work_cv.wait(lock, [this] {
                return !_work_queue.empty() || _shutdown;
            });

            if (_shutdown) break;

            if (_work_queue.empty()) continue;

            work_item = _work_queue.front();
            _work_queue.pop();
        }

        try {
            // Load image (this is the expensive operation)
            auto image_tensor = load_image(work_item.cam_info);

            // Create camera
            Camera camera(work_item.cam_info._camera_ID,
                          work_item.cam_info._R,
                          work_item.cam_info._T,
                          work_item.cam_info._fov_x,
                          work_item.cam_info._fov_y,
                          std::move(image_tensor),
                          work_item.cam_info._image_name,
                          work_item.camera_idx);

            // Add to ready queue
            {
                std::lock_guard<std::mutex> lock(_ready_mutex);
                _ready_queue.emplace(std::move(camera), work_item.epoch, work_item.camera_idx);
            }
            _ready_cv.notify_one();

        } catch (const std::exception& e) {
            std::cerr << "Worker " << worker_id << " error loading camera "
                      << work_item.camera_idx << ": " << e.what() << std::endl;
        }
    }

    std::cout << "Worker " << worker_id << " stopped" << std::endl;
}

void DataLoader::fill_work_queue() {
    std::lock_guard<std::mutex> lock(_work_mutex);

    // Fill work queue with prefetch_factor * num_workers items
    int items_to_add = _config.prefetch_factor * _config.num_workers;

    for (int i = 0; i < items_to_add && _current_idx < _indices.size(); ++i) {
        int idx = _current_idx++;
        int camera_idx = _indices[idx];

        WorkItem work_item;
        work_item.camera_idx = camera_idx;
        work_item.epoch = _current_epoch;
        work_item.cam_info = _camera_infos[camera_idx];

        _work_queue.push(work_item);
    }

    _work_cv.notify_all();
}

Camera DataLoader::get_next_camera() {
    // If no workers, load synchronously
    if (_config.num_workers == 0) {
        if (_current_idx >= _indices.size()) {
            prepare_next_epoch();
        }

        int idx = _current_idx++;
        int camera_idx = _indices[idx];
        const auto& cam_info = _camera_infos[camera_idx];

        auto image_tensor = load_image(cam_info);
        return Camera(cam_info._camera_ID, cam_info._R, cam_info._T,
                      cam_info._fov_x, cam_info._fov_y,
                      std::move(image_tensor), cam_info._image_name, camera_idx);
    }

    // Multi-threaded path
    LoadedCamera loaded_camera = [this]() {
        std::unique_lock<std::mutex> lock(_ready_mutex);

        // Wait for a camera to be ready
        _ready_cv.wait(lock, [this] {
            return !_ready_queue.empty() || _shutdown;
        });

        if (_shutdown) {
            throw std::runtime_error("DataLoader is shutting down");
        }

        auto camera = std::move(_ready_queue.front());
        _ready_queue.pop();
        return camera;
    }();

    // Refill work queue if needed
    fill_work_queue();

    // Check if we need to start next epoch
    if (_current_idx >= _indices.size()) {
        prepare_next_epoch();
    }

    return std::move(loaded_camera.camera);
}

void DataLoader::prepare_next_epoch() {
    _current_epoch++;
    _current_idx = 0;

    if (_config.shuffle) {
        std::shuffle(_indices.begin(), _indices.end(), _rng);
    }

    std::cout << "Starting epoch " << _current_epoch << std::endl;

    // Refill work queue for new epoch
    if (_config.num_workers > 0) {
        fill_work_queue();
    }
}

bool DataLoader::has_more_data() const {
    if (_config.num_workers == 0) {
        return _current_idx < _indices.size();
    }

    // With workers, check both work queue and ready queue
    std::lock_guard<std::mutex> work_lock(_work_mutex);
    std::lock_guard<std::mutex> ready_lock(_ready_mutex);
    return !_work_queue.empty() || !_ready_queue.empty() || _current_idx < _indices.size();
}

void DataLoader::reset() {
    // Clear queues
    if (_config.num_workers > 0) {
        {
            std::lock_guard<std::mutex> work_lock(_work_mutex);
            std::queue<WorkItem> empty_work;
            _work_queue.swap(empty_work);
        }
        {
            std::lock_guard<std::mutex> ready_lock(_ready_mutex);
            std::queue<LoadedCamera> empty_ready;
            _ready_queue.swap(empty_ready);
        }
    }

    prepare_next_epoch();
}

torch::Tensor DataLoader::load_image(const CameraInfo& cam_info) {
    // Load image on demand
    auto [img_data, width, height, channels] = read_image(cam_info._image_path, _params.resolution);

    // Create tensor from image data
    torch::Tensor image_tensor = torch::from_blob(img_data,
                                                  {height, width, channels},
                                                  {width * channels, channels, 1},
                                                  torch::kUInt8);

    // Convert to float and normalize
    image_tensor = image_tensor.to(torch::kFloat32).permute({2, 0, 1}).clone() / 255.0f;

    // Pin memory for faster GPU transfer if requested
    if (_config.pin_memory && torch::cuda::is_available()) {
        image_tensor = image_tensor.pin_memory();
    }

    // Free the image data
    free_image(img_data);

    return image_tensor;
}

//===================================================================
// Factory function with worker support
std::unique_ptr<DataLoader> create_dataloader(const gs::param::ModelParameters& params,
                                              int num_workers,
                                              bool pin_memory) {
    if (!std::filesystem::exists(params.source_path)) {
        throw std::runtime_error("Data path does not exist: " + params.source_path.string());
    }

    auto scene_info = read_colmap_scene_info(params.source_path, params.resolution);

    DataLoader::Config config;
    config.shuffle = true;
    config.num_workers = num_workers;
    config.prefetch_factor = 2;
    config.pin_memory = pin_memory;
    config.persistent_workers = true;

    return std::make_unique<DataLoader>(std::move(scene_info), params, config);
}