/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "dataset.hpp"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <thread>

namespace gs::training {

    // Internal efficient implementation for training
    class EfficientDataLoader {
    public:
        EfficientDataLoader(
            std::shared_ptr<CameraDataset> dataset,
            int num_workers);

        ~EfficientDataLoader();

        CameraWithImage get_next();

    private:
        std::shared_ptr<CameraDataset> dataset_;
        int num_workers_;

        // Pre-allocated GPU buffer pool
        struct BufferSlot {
            torch::Tensor gpu_buffer;
            std::atomic<bool> in_use{false};
            int last_width = 0;
            int last_height = 0;
        };
        std::vector<std::unique_ptr<BufferSlot>> buffer_pool_;

        // Worker threads
        std::vector<std::thread> workers_;
        std::atomic<bool> should_stop_{false};

        // Ready queue with bounded size
        std::deque<CameraWithImage> ready_queue_;
        mutable std::mutex queue_mutex_;
        std::condition_variable queue_cv_;
        std::condition_variable space_cv_;

        // Random sampling
        std::vector<size_t> indices_;
        std::atomic<size_t> next_index_{0};
        std::mt19937 rng_{std::random_device{}()};
        std::mutex index_mutex_;

        void worker_thread(int worker_id);
        BufferSlot* acquire_buffer();
        void release_buffer(BufferSlot* buffer);
    };

    // Wrapper for infinite training dataloader
    class InfiniteDataLoaderWrapper {
        std::unique_ptr<EfficientDataLoader> loader_;
        CameraWithImage current_;

    public:
        InfiniteDataLoaderWrapper(std::unique_ptr<EfficientDataLoader> loader);

        struct Iterator {
            InfiniteDataLoaderWrapper* parent;

            CameraWithImage operator*() const { return parent->current_; }
            Iterator& operator++() {
                parent->current_ = parent->loader_->get_next();
                return *this;
            }
            bool operator!=(const Iterator&) const { return true; }
        };

        Iterator begin() {
            current_ = loader_->get_next();
            return Iterator{this};
        }
    };

    // Simple evaluation dataloader
    class EvalDataLoader {
    public:
        explicit EvalDataLoader(std::shared_ptr<CameraDataset> dataset);

        struct Iterator {
            EvalDataLoader* parent;
            size_t index;

            CameraWithImage operator*() const;
            Iterator& operator++();
            bool operator!=(const Iterator& other) const;
        };

        Iterator begin();
        Iterator end();

    private:
        std::shared_ptr<CameraDataset> dataset_;
        size_t dataset_size_;
    };

    // Factory functions
    std::unique_ptr<InfiniteDataLoaderWrapper>
    create_efficient_infinite_dataloader(
        std::shared_ptr<CameraDataset> dataset,
        int num_workers);

    std::unique_ptr<EvalDataLoader>
    create_eval_dataloader(
        std::shared_ptr<CameraDataset> dataset);

} // namespace gs::training