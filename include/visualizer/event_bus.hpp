#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <typeindex>
#include <unordered_map>
#include <vector>

namespace gs {

    /**
     * @brief Type-safe event channel for publishing/subscribing to specific event types
     *
     * Uses composition to handle event delivery without inheritance
     */
    template <typename EventType>
    class EventChannel {
    public:
        using Handler = std::function<void(const EventType&)>;
        using HandlerId = size_t;

        /**
         * @brief Subscribe to events on this channel
         * @param handler Function to call when event is published
         * @return Handler ID for later unsubscription
         */
        HandlerId subscribe(Handler handler) {
            std::lock_guard<std::mutex> lock(mutex_);
            HandlerId id = next_id_++;
            handlers_[id] = std::move(handler);
            return id;
        }

        /**
         * @brief Unsubscribe a handler
         * @param id Handler ID returned from subscribe
         */
        void unsubscribe(HandlerId id) {
            std::lock_guard<std::mutex> lock(mutex_);
            handlers_.erase(id);
        }

        /**
         * @brief Publish an event to all subscribers
         * @param event Event to publish
         */
        void publish(const EventType& event) {
            std::vector<Handler> handlers_copy;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                handlers_copy.reserve(handlers_.size());
                for (const auto& [id, handler] : handlers_) {
                    handlers_copy.push_back(handler);
                }
            }

            // Call handlers outside of lock to prevent deadlocks
            for (const auto& handler : handlers_copy) {
                handler(event);
            }
        }

        /**
         * @brief Clear all subscribers
         */
        void clear() {
            std::lock_guard<std::mutex> lock(mutex_);
            handlers_.clear();
        }

        /**
         * @brief Get number of subscribers
         */
        size_t subscriber_count() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return handlers_.size();
        }

    private:
        mutable std::mutex mutex_;
        std::unordered_map<HandlerId, Handler> handlers_;
        HandlerId next_id_ = 1;
    };

    /**
     * @brief Central event bus for application-wide event distribution
     *
     * Provides type-safe channels for different event types without requiring
     * inheritance from a base event class.
     */
    class EventBus {
    public:
        /**
         * @brief Get or create a channel for a specific event type
         * @tparam EventType Type of event
         * @return Shared pointer to the event channel
         */
        template <typename EventType>
        std::shared_ptr<EventChannel<EventType>> channel() {
            std::lock_guard<std::mutex> lock(mutex_);

            const auto type_id = std::type_index(typeid(EventType));
            auto it = channels_.find(type_id);

            if (it != channels_.end()) {
                return std::static_pointer_cast<EventChannel<EventType>>(it->second);
            }

            auto new_channel = std::make_shared<EventChannel<EventType>>();
            channels_[type_id] = new_channel;
            return new_channel;
        }

        /**
         * @brief Convenience method to publish an event
         * @tparam EventType Type of event
         * @param event Event to publish
         */
        template <typename EventType>
        void publish(const EventType& event) {
            channel<EventType>()->publish(event);
        }

        /**
         * @brief Convenience method to subscribe to an event
         * @tparam EventType Type of event
         * @param handler Handler function
         * @return Handler ID for unsubscription
         */
        template <typename EventType>
        typename EventChannel<EventType>::HandlerId subscribe(
            typename EventChannel<EventType>::Handler handler) {
            return channel<EventType>()->subscribe(std::move(handler));
        }

        /**
         * @brief Clear all channels
         */
        void clear() {
            std::lock_guard<std::mutex> lock(mutex_);
            channels_.clear();
        }

    private:
        mutable std::mutex mutex_;
        std::unordered_map<std::type_index, std::shared_ptr<void>> channels_;
    };

} // namespace gs
