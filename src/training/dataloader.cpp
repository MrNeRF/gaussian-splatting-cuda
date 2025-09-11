/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "dataloader.hpp"
#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include <algorithm>
#include <c10/cuda/CUDAGuard.h>

namespace gs::training {

    // =============================================================================
    // Efficient Training DataLoader Implementation
    // =============================================================================

    EfficientDataLoader::EfficientDataLoader(
        std::shared_ptr<CameraDataset> dataset,
        int num_workers)
        : dataset_(dataset),
          num_workers_(num_workers) {

        // Get the actual dataset size (respects train/val split)
        size_t dataset_size = dataset_->size();

        // Initialize indices for the dataset (not all cameras!)
        indices_.resize(dataset_size);
        std::iota(indices_.begin(), indices_.end(), 0);
        std::shuffle(indices_.begin(), indices_.end(), rng_);

        // Pre-allocate GPU buffers - this is THE key optimization
        // We allocate num_workers * 2 buffers for double buffering
        const size_t buffer_count = num_workers * 2;
        buffer_pool_.reserve(buffer_count);

        LOG_INFO("Pre-allocating {} GPU buffers for efficient dataloader", buffer_count);
        LOG_INFO("Dataset size: {} images", dataset_size);

        // Estimate size from first image
        if (dataset_size > 0) {
            // Get first camera through the dataset to respect the split
            auto first_example = dataset_->get(0);
            Camera* first_camera = first_example.camera;

            auto [w, h, c] = get_image_info(first_camera->image_path());
            int resize = dataset_->get_resize_factor();
            if (resize > 1) {
                w /= resize;
                h /= resize;
            }

            // Pre-allocate GPU tensors
            for (size_t i = 0; i < buffer_count; ++i) {
                auto slot = std::make_unique<BufferSlot>();
                // Allocate GPU memory once
                slot->gpu_buffer = torch::empty(
                    {c, h, w}, // CHW format
                    torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(torch::kCUDA));
                slot->last_width = w;
                slot->last_height = h;
                buffer_pool_.push_back(std::move(slot));
            }

            size_t total_vram = buffer_count * w * h * c * sizeof(float);
            LOG_INFO("Total VRAM pre-allocated for dataloader: {:.2f} MB",
                     total_vram / (1024.0 * 1024.0));
        }

        // Start worker threads
        for (int i = 0; i < num_workers_; ++i) {
            workers_.emplace_back(&EfficientDataLoader::worker_thread, this, i);
        }

        LOG_DEBUG("Started {} dataloader worker threads", num_workers_);
    }

    EfficientDataLoader::~EfficientDataLoader() {
        should_stop_ = true;
        queue_cv_.notify_all();
        space_cv_.notify_all();

        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }

        LOG_DEBUG("Stopped all dataloader worker threads");
    }

    EfficientDataLoader::BufferSlot* EfficientDataLoader::acquire_buffer() {
        // Find an unused buffer
        while (!should_stop_) {
            for (auto& slot : buffer_pool_) {
                bool expected = false;
                if (slot->in_use.compare_exchange_strong(expected, true)) {
                    return slot.get();
                }
            }
            // If no buffer available, wait a bit
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        return nullptr;
    }

    void EfficientDataLoader::release_buffer(BufferSlot* buffer) {
        if (buffer) {
            buffer->in_use = false;
        }
    }

    void EfficientDataLoader::worker_thread(int worker_id) {
        const int resize_factor = dataset_->get_resize_factor();

        // Create a dedicated CUDA stream for this worker
        at::cuda::CUDAStream stream = at::cuda::getStreamFromPool(false);

        while (!should_stop_) {
            // Check if queue has space
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                space_cv_.wait(lock, [this] {
                    return ready_queue_.size() < static_cast<size_t>(num_workers_ * 2) || should_stop_;
                });

                if (should_stop_)
                    break;
            }

            // Get next index
            size_t dataset_idx;
            {
                std::lock_guard<std::mutex> lock(index_mutex_);
                if (next_index_ >= indices_.size()) {
                    // Reshuffle for new epoch
                    std::shuffle(indices_.begin(), indices_.end(), rng_);
                    next_index_ = 0;
                    LOG_TRACE("Worker {} reshuffled dataset for new epoch", worker_id);
                }
                dataset_idx = indices_[next_index_++];
            }

            // This ensures we only get images that belong to this dataset's split
            auto dataset_example = dataset_->get(dataset_idx);
            Camera* camera = dataset_example.camera;

            // Acquire a buffer slot
            BufferSlot* slot = acquire_buffer();
            if (!slot)
                continue;

            // Load image data using image_io's standard function
            auto [data, w, h, c] = load_image(camera->image_path(), resize_factor);

            // Update camera dimensions
            camera->update_image_dimensions(w, h);

            // Check if buffer needs resize
            if (slot->last_width != w || slot->last_height != h) {
                LOG_DEBUG("Resizing buffer from {}x{} to {}x{}",
                          slot->last_width, slot->last_height, w, h);
                slot->gpu_buffer = torch::empty(
                    {c, h, w},
                    torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(torch::kCUDA));
                slot->last_width = w;
                slot->last_height = h;
            }

            // Use stream for this worker
            c10::cuda::CUDAStreamGuard guard(stream);

            // Transfer to GPU with minimal allocations
            // Create pinned tensor WITHOUT copying - just wrapping the existing data
            auto pinned = torch::from_blob(
                data,
                {h, w, c},
                torch::TensorOptions()
                    .dtype(torch::kUInt8)
                    .pinned_memory(true));

            // Direct conversion and copy into pre-allocated buffer
            // This is the KEY - we reuse slot->gpu_buffer!
            slot->gpu_buffer.copy_(
                pinned.to(torch::kCUDA, /*non_blocking=*/true)
                    .permute({2, 0, 1}) // HWC -> CHW
                    .to(torch::kFloat32)
                    .div_(255.0f),
                /*non_blocking=*/true);

            // Synchronize this stream
            stream.synchronize();

            // Free CPU memory using image_io's function
            free_image(data);

            // Create the example - clone() here is cheap as it just increments ref count
            CameraWithImage example{camera, slot->gpu_buffer.clone()};

            // Release buffer back to pool
            release_buffer(slot);

            // Add to ready queue
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                ready_queue_.push_back(std::move(example));
            }
            queue_cv_.notify_one();
        }

        LOG_TRACE("Worker {} thread exiting", worker_id);
    }

    CameraWithImage EfficientDataLoader::get_next() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this] {
            return !ready_queue_.empty() || should_stop_;
        });

        if (should_stop_ && ready_queue_.empty()) {
            throw std::runtime_error("DataLoader stopped");
        }

        CameraWithImage example = std::move(ready_queue_.front());
        ready_queue_.pop_front();

        // Signal that there's space in queue
        space_cv_.notify_one();

        return example;
    }

    InfiniteDataLoaderWrapper::InfiniteDataLoaderWrapper(
        std::unique_ptr<EfficientDataLoader> loader)
        : loader_(std::move(loader)) {}

    std::unique_ptr<InfiniteDataLoaderWrapper>
    create_efficient_infinite_dataloader(
        std::shared_ptr<CameraDataset> dataset,
        int num_workers) {

        auto loader = std::make_unique<EfficientDataLoader>(dataset, num_workers);
        return std::make_unique<InfiniteDataLoaderWrapper>(std::move(loader));
    }

    // =============================================================================
    // Simple Evaluation DataLoader Implementation
    // =============================================================================

    EvalDataLoader::EvalDataLoader(std::shared_ptr<CameraDataset> dataset)
        : dataset_(dataset),
          dataset_size_(dataset->size()) {
        LOG_DEBUG("Created evaluation dataloader with {} images", dataset_size_);
    }

    CameraWithImage EvalDataLoader::Iterator::operator*() const {
        // Simply get the example from the dataset (uses Camera::load_and_get_image)
        return parent->dataset_->get(index);
    }

    EvalDataLoader::Iterator& EvalDataLoader::Iterator::operator++() {
        ++index;
        return *this;
    }

    bool EvalDataLoader::Iterator::operator!=(const Iterator& other) const {
        return index != other.index;
    }

    EvalDataLoader::Iterator EvalDataLoader::begin() {
        return Iterator{this, 0};
    }

    EvalDataLoader::Iterator EvalDataLoader::end() {
        return Iterator{this, dataset_size_};
    }

    std::unique_ptr<EvalDataLoader> create_eval_dataloader(
        std::shared_ptr<CameraDataset> dataset) {
        return std::make_unique<EvalDataLoader>(dataset);
    }

} // namespace gs::training