#pragma once

#include "visualizer/event_bus.hpp"
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <optional>

namespace gs {

    /**
     * @brief Utility for handling request/response event patterns
     *
     * This helper makes it easy to send a request event and wait for a response
     */
    template <typename RequestType, typename ResponseType>
    class EventResponseHandler {
    public:
        explicit EventResponseHandler(std::shared_ptr<EventBus> event_bus)
            : event_bus_(event_bus) {

            // Subscribe to responses
            response_handler_id_ = event_bus_->subscribe<ResponseType>(
                [this](const ResponseType& response) {
                    handleResponse(response);
                });
        }

        ~EventResponseHandler() {
            if (event_bus_ && response_handler_id_ > 0) {
                event_bus_->channel<ResponseType>()->unsubscribe(response_handler_id_);
            }
        }

        /**
         * @brief Send request and wait for response
         * @param request The request to send
         * @param timeout_ms Timeout in milliseconds (0 = no timeout)
         * @return Response if received within timeout
         */
        std::optional<ResponseType> querySync(
            const RequestType& request,
            std::chrono::milliseconds timeout = std::chrono::milliseconds(100)) {

            // Clear any previous response
            {
                std::lock_guard<std::mutex> lock(mutex_);
                last_response_.reset();
                response_received_ = false;
            }

            // Send request
            event_bus_->publish(request);

            // Wait for response
            std::unique_lock<std::mutex> lock(mutex_);
            if (timeout.count() > 0) {
                if (!cv_.wait_for(lock, timeout, [this] { return response_received_; })) {
                    return std::nullopt; // Timeout
                }
            } else {
                cv_.wait(lock, [this] { return response_received_; });
            }

            return last_response_;
        }

        /**
         * @brief Send request and get future for response
         * @param request The request to send
         * @return Future that will contain the response
         */
        std::future<ResponseType> queryAsync(const RequestType& request) {
            auto promise = std::make_shared<std::promise<ResponseType>>();
            auto future = promise->get_future();

            // Subscribe to single response
            auto handler_id = event_bus_->subscribe<ResponseType>(
                [promise, this](const ResponseType& response) {
                    promise->set_value(response);
                    // Note: Can't unsubscribe from within handler
                });

            // Send request
            event_bus_->publish(request);

            return future;
        }

    private:
        void handleResponse(const ResponseType& response) {
            std::lock_guard<std::mutex> lock(mutex_);
            last_response_ = response;
            response_received_ = true;
            cv_.notify_all();
        }

        std::shared_ptr<EventBus> event_bus_;
        size_t response_handler_id_ = 0;

        std::mutex mutex_;
        std::condition_variable cv_;
        std::optional<ResponseType> last_response_;
        bool response_received_ = false;
    };

} // namespace gs
