#pragma once
#include "core/event_bus.hpp"
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <glm/glm.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace gs {

    // Forward declarations
    class Scene;

// Clean event macro - uses gs::event::bus()
#define EVENT(Name, ...)                                   \
    struct Name {                                          \
        using event_id = Name;                             \
        __VA_ARGS__                                        \
                                                           \
        void emit() const {                                \
            ::gs::event::bus().emit(*this);                \
        }                                                  \
                                                           \
        static auto when(auto&& handler) {                 \
            return ::gs::event::bus().when<Name>(          \
                std::forward<decltype(handler)>(handler)); \
        }                                                  \
    }

    namespace events {

        // ============================================================================
        // Commands - User actions that request something to happen
        // ============================================================================
        namespace cmd {
            EVENT(StartTraining, );
            EVENT(PauseTraining, );
            EVENT(ResumeTraining, );
            EVENT(StopTraining, );
            EVENT(SaveCheckpoint, std::optional<int> iteration;);
            EVENT(LoadFile, std::filesystem::path path; bool is_dataset;);
            EVENT(ClearScene, );
            EVENT(ResetCamera, );
            EVENT(ShowWindow, std::string window_name; bool show;);
            EVENT(ExecuteConsole, std::string command;);
            EVENT(GoToCamView, int cam_id;);
        } // namespace cmd

        // ============================================================================
        // Tools - Notifications about what has happened
        // ============================================================================

        namespace tools {
            EVENT(ToolEnabled, std::string tool_name;);
            EVENT(ToolDisabled, std::string tool_name;);
            EVENT(CropBoxSettingsChanged,
                  bool show_box;
                  bool use_box;);
            EVENT(AxesSettingsChanged,
                  bool show_axes;);
        } // namespace tools
        // ============================================================================
        // State - Notifications about what has happened
        // ============================================================================
        namespace state {
            // Training state
            EVENT(TrainingStarted, int total_iterations;);
            EVENT(TrainingProgress, int iteration; float loss; int num_gaussians; bool is_refining = false;);
            EVENT(TrainingPaused, int iteration;);
            EVENT(TrainingResumed, int iteration;);
            EVENT(TrainingCompleted, int iteration; float final_loss; bool success; std::optional<std::string> error;);
            EVENT(TrainingStopped, int iteration; bool user_requested;);

            // Scene state
            EVENT(SceneLoaded,
                  Scene* scene;
                  std::filesystem::path path;
                  enum class Type{PLY, Dataset} type;
                  size_t num_gaussians;);
            EVENT(SceneCleared, );
            EVENT(ModelUpdated, int iteration; size_t num_gaussians;);
            EVENT(SceneChanged, );

            // Data loading
            EVENT(DatasetLoadStarted, std::filesystem::path path;);
            EVENT(DatasetLoadProgress, std::filesystem::path path; float progress; std::string step;);
            EVENT(DatasetLoadCompleted,
                  std::filesystem::path path;
                  bool success;
                  std::optional<std::string> error;
                  size_t num_images;
                  size_t num_points;);

            // Evaluation
            EVENT(EvaluationStarted, int iteration; size_t num_images;);
            EVENT(EvaluationProgress, int iteration; size_t current; size_t total;);
            EVENT(EvaluationCompleted,
                  int iteration;
                  float psnr;
                  float ssim;
                  float lpips;
                  float elapsed_time;
                  int num_gaussians;);

            // System state
            EVENT(CheckpointSaved, int iteration; std::filesystem::path path;);
            EVENT(MemoryUsage,
                  size_t gpu_used;
                  size_t gpu_total;
                  float gpu_percent;
                  size_t ram_used;
                  size_t ram_total;
                  float ram_percent;);
            EVENT(FrameRendered, float render_ms; float fps; int num_gaussians;);
        } // namespace state

        // ============================================================================
        // Notifications - Important messages for the user
        // ============================================================================
        namespace notify {
            EVENT(Info, std::string message; int duration_ms = 3000;);
            EVENT(Success, std::string message; int duration_ms = 3000;);
            EVENT(Warning, std::string message; int duration_ms = 5000;);
            EVENT(Error, std::string message; std::string details = ""; int duration_ms = 0;);

            EVENT(MemoryWarning,
                  enum class Type{GPU, RAM} type;
                  float usage_percent;
                  std::string message;);

            EVENT(Log,
                  enum class Level{Debug, Info, Warning, Error} level;
                  std::string message;
                  std::string source = "";);
        } // namespace notify

        // ============================================================================
        // UI - User interface updates
        // ============================================================================
        namespace ui {
            EVENT(WindowResized, int width; int height;);
            EVENT(CameraMove, glm::mat3 rotation; glm::vec3 translation;);
            EVENT(SpeedChanged, float current_speed; float max_speed;);
            EVENT(RenderSettingsChanged,
                  std::optional<float> fov;
                  std::optional<float> scaling_modifier;
                  std::optional<bool> antialiasing;
                  std::optional<glm::vec3> background_color;);
            EVENT(RenderModeChanged, std::string old_mode; std::string new_mode;);
            EVENT(PointCloudModeChanged, bool enabled; float voxel_size;);
            EVENT(NodeSelected,
                  std::string path;
                  std::string type;
                  std::unordered_map<std::string, std::string> metadata;);
            EVENT(CropBoxChanged,
                  glm::vec3 min_bounds;
                  glm::vec3 max_bounds;
                  bool enabled;);
            EVENT(CropBoxVisibilityChanged, bool visible;);
            EVENT(ConsoleResult, std::string command; std::string result;);
        } // namespace ui

        // ============================================================================
        // Queries - Synchronous request/response for information
        // ============================================================================
        namespace query {
            // Trainer queries
            EVENT(GetTrainerState, );
            EVENT(TrainerState,
                  enum class State{Idle, Ready, Running, Paused, Completed, Error} state;
                  int current_iteration;
                  float current_loss;
                  int total_iterations;
                  std::optional<std::string> error_message;);

            // Scene queries
            EVENT(GetSceneInfo, );
            EVENT(SceneInfo,
                  enum class Type{None, PLY, Dataset} type;
                  std::filesystem::path source_path;
                  size_t num_gaussians;
                  bool is_training;
                  bool has_model;);

            // Model queries
            EVENT(GetModelInfo, );
            EVENT(ModelInfo,
                  bool has_model;
                  size_t num_gaussians;
                  int sh_degree;
                  float scene_scale;
                  std::string source;);

            // Render queries
            EVENT(GetRenderCapabilities, );
            EVENT(RenderCapabilities,
                  std::vector<std::string> modes;
                  bool supports_antialiasing;
                  bool supports_depth;
                  int max_width;
                  int max_height;);
        } // namespace query

        // ============================================================================
        // Internal - System coordination events (not for external use)
        // ============================================================================
        namespace internal {
            EVENT(TrainerReady, );
            EVENT(TrainingReadyToStart, );
            EVENT(WindowFocusLost, );
            EVENT(RenderRequest,
                  glm::mat3 view_rotation;
                  glm::vec3 view_translation;
                  glm::ivec2 viewport_size;
                  float fov;
                  float scaling_modifier;
                  bool antialiasing;
                  int render_mode;
                  void* crop_box;
                  size_t request_id;);
            EVENT(RenderComplete, size_t request_id; bool success; float render_ms;);
        } // namespace internal

    } // namespace events

    // ============================================================================
    // Helper for Query/Response pattern
    // ============================================================================
    template <typename Request, typename Response>
    class Query {
        std::optional<Response> response_;
        std::condition_variable cv_;
        std::mutex mutex_;

    public:
        Response send(const Request& req, std::chrono::milliseconds timeout = std::chrono::milliseconds(100)) {
            // Subscribe to response
            [[maybe_unused]] auto handler = Response::when([this](const Response& r) {
                std::lock_guard lock(mutex_);
                response_ = r;
                cv_.notify_one();
            });

            // Send request
            req.emit();

            // Wait for response
            std::unique_lock lock(mutex_);
            if (!cv_.wait_for(lock, timeout, [this] { return response_.has_value(); })) {
                throw std::runtime_error("Query timeout");
            }

            return *response_;
        }
    };

    // ============================================================================
    // Convenience functions
    // ============================================================================
    template <event::Event E>
    inline void emit(const E& event) {
        event.emit();
    }

    template <event::Event E>
    inline auto when(auto&& handler) {
        return E::when(std::forward<decltype(handler)>(handler));
    }

} // namespace gs