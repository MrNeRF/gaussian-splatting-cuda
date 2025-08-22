#pragma once

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <future>
#include <list>
#include <mutex>
#include <queue>
#include <thread>

namespace gs::loader {

    /**
     * @brief Thread-safe queue for managing asynchronous loading operations
     */
    class LoadingQueue {
    public:
        struct Task {
            std::filesystem::path path;
            std::function<void()> work;
            std::promise<void> completion;
        };

        explicit LoadingQueue(size_t num_workers = std::thread::hardware_concurrency());
        ~LoadingQueue();

        // Delete copy operations
        LoadingQueue(const LoadingQueue&) = delete;
        LoadingQueue& operator=(const LoadingQueue&) = delete;

        /**
         * @brief Enqueue a loading task
         * @param path Path being loaded (for tracking)
         * @param work Function to execute
         * @return Future that completes when task is done
         */
        std::future<void> enqueue(const std::filesystem::path& path,
                                  std::function<void()> work);

        /**
         * @brief Cancel all pending tasks
         */
        void cancelAll();

        /**
         * @brief Get number of pending tasks
         */
        size_t pendingCount() const {
            std::lock_guard lock(mutex_);
            return tasks_.size();
        }

        /**
         * @brief Check if queue is empty
         */
        bool empty() const {
            std::lock_guard lock(mutex_);
            return tasks_.empty();
        }

        /**
         * @brief Wait for all tasks to complete
         */
        void waitAll();

    private:
        void workerThread();

    private:
        mutable std::mutex mutex_;
        std::condition_variable cv_;
        std::queue<std::unique_ptr<Task>> tasks_;
        std::vector<std::thread> workers_;
        std::atomic<bool> stop_{false};
        std::atomic<size_t> active_tasks_{0};
    };

    // ============================================================================
    // Loading Cache (optional, for future use)
    // ============================================================================

    /**
     * @brief Simple LRU cache for loaded data
     */
    template <typename T>
    class LoadingCache {
    public:
        explicit LoadingCache(size_t max_size = 10) : max_size_(max_size) {}

        void put(const std::filesystem::path& key, std::shared_ptr<T> value) {
            std::lock_guard lock(mutex_);

            // Remove if already exists
            auto it = cache_map_.find(key);
            if (it != cache_map_.end()) {
                cache_list_.erase(it->second);
                cache_map_.erase(it);
            }

            // Add to front
            cache_list_.push_front({key, value});
            cache_map_[key] = cache_list_.begin();

            // Evict if necessary
            while (cache_list_.size() > max_size_) {
                auto last = cache_list_.end();
                --last;
                cache_map_.erase(last->first);
                cache_list_.pop_back();
            }
        }

        std::shared_ptr<T> get(const std::filesystem::path& key) {
            std::lock_guard lock(mutex_);

            auto it = cache_map_.find(key);
            if (it == cache_map_.end()) {
                return nullptr;
            }

            // Move to front
            cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
            return it->second->second;
        }

        void clear() {
            std::lock_guard lock(mutex_);
            cache_list_.clear();
            cache_map_.clear();
        }

        size_t size() const {
            std::lock_guard lock(mutex_);
            return cache_list_.size();
        }

    private:
        using CacheItem = std::pair<std::filesystem::path, std::shared_ptr<T>>;
        using CacheList = std::list<CacheItem>;
        using CacheMap = std::unordered_map<std::filesystem::path, typename CacheList::iterator>;

        mutable std::mutex mutex_;
        CacheList cache_list_;
        CacheMap cache_map_;
        size_t max_size_;
    };

} // namespace gs::loader