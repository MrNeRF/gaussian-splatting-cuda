/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/command_processor.hpp"
#include "core/splat_data.hpp"
#include "scene/scene_manager.hpp"
#include "training/training_manager.hpp"
#include <iomanip>
#include <sstream>
#include <torch/torch.h>

namespace gs {

    CommandProcessor::CommandProcessor(SceneManager* scene_manager)
        : scene_manager_(scene_manager) {
        registerBuiltinCommands();
    }

    void CommandProcessor::registerBuiltinCommands() {
        registerCommand("help", [this]() { return handleHelp(); });
        registerCommand("h", [this]() { return handleHelp(); });
        registerCommand("status", [this]() { return handleStatus(); });
        registerCommand("model_info", [this]() { return handleModelInfo(); });
        registerCommand("gpu_info", [this]() { return handleGpuInfo(); });
        registerCommand("clear", []() { return ""; }); // Handled by console
    }

    std::string CommandProcessor::processCommand(const std::string& command) {
        if (command.empty()) {
            return "";
        }

        // Check for tensor_info command with parameter
        if (command.substr(0, 11) == "tensor_info") {
            std::string tensor_name = "";
            if (command.length() > 12) {
                tensor_name = command.substr(12);
            }
            return handleTensorInfo(tensor_name);
        }

        // Look up command in registry
        auto it = commands_.find(command);
        if (it != commands_.end()) {
            return it->second();
        }

        return "Unknown command: '" + command + "'. Type 'help' for available commands.";
    }

    void CommandProcessor::registerCommand(const std::string& name, CommandHandler handler) {
        commands_[name] = handler;
    }

    std::string CommandProcessor::handleHelp() {
        std::ostringstream result;
        result << "Available commands:\n";
        result << "  help, h - Show this help\n";
        result << "  clear - Clear console\n";
        result << "  status - Show training status\n";
        result << "  model_info - Show model information\n";
        result << "  tensor_info <name> - Show tensor information\n";
        result << "  gpu_info - Show GPU information\n";
        return result.str();
    }

    std::string CommandProcessor::handleStatus() {
        std::ostringstream result;

        if (scene_manager_) {
            // Check content type first
            auto content_type = scene_manager_->getContentType();

            if (content_type == SceneManager::ContentType::Dataset) {
                // Only check training state if we have a dataset loaded
                auto* trainer_manager = scene_manager_->getTrainerManager();
                if (trainer_manager && trainer_manager->hasTrainer()) {
                    auto state = trainer_manager->getState();

                    result << "Training Status:\n";
                    result << "  State: ";

                    // Convert state to string
                    switch (state) {
                    case TrainerManager::State::Idle:
                        result << "Idle";
                        break;
                    case TrainerManager::State::Ready:
                        result << "Ready";
                        break;
                    case TrainerManager::State::Running:
                        result << "Running";
                        break;
                    case TrainerManager::State::Paused:
                        result << "Paused";
                        break;
                    case TrainerManager::State::Stopping:
                        result << "Stopping";
                        break;
                    case TrainerManager::State::Completed:
                        result << "Completed";
                        break;
                    case TrainerManager::State::Error:
                        result << "Error";
                        break;
                    default:
                        result << "Unknown";
                        break;
                    }

                    result << "\n";
                    result << "  Current Iteration: " << trainer_manager->getCurrentIteration() << "\n";
                    result << "  Current Loss: " << std::fixed << std::setprecision(6)
                           << trainer_manager->getCurrentLoss();

                    auto error_msg = trainer_manager->getLastError();
                    if (!error_msg.empty()) {
                        result << "\n  Error: " << error_msg;
                    }
                } else {
                    result << "Dataset loaded but no trainer available";
                }
            } else if (content_type == SceneManager::ContentType::SplatFiles) {
                result << "Splat Viewer mode (no training)";
            } else {
                result << "No content loaded";
            }
        } else {
            result << "No scene manager available";
        }

        return result.str();
    }

    std::string CommandProcessor::handleModelInfo() {
        std::ostringstream result;

        if (scene_manager_) {
            auto info = scene_manager_->getSceneInfo();

            if (info.has_model) {
                result << "Scene Information:\n";
                result << "  Type: " << info.source_type << "\n";
                result << "  Source: " << info.source_path.filename().string() << "\n";
                result << "  Number of Gaussians: " << info.num_gaussians << "\n";

                if (info.source_type == "PLY") {
                    result << "  Number of Nodes: " << info.num_nodes << "\n";
                }

                // Only mention training if we have a dataset loaded
                if (scene_manager_->getContentType() == SceneManager::ContentType::Dataset) {
                    auto* trainer_manager = scene_manager_->getTrainerManager();
                    if (trainer_manager && trainer_manager->isRunning()) {
                        result << "  Training Mode: Active\n";
                    }
                }
            } else {
                result << "No scene loaded";
            }
        } else {
            result << "No scene manager available";
        }

        return result.str();
    }

    std::string CommandProcessor::handleGpuInfo() {
        std::ostringstream result;

        size_t free_byte, total_byte;
        cudaDeviceSynchronize();
        cudaMemGetInfo(&free_byte, &total_byte);

        double free_gb = free_byte / 1024.0 / 1024.0 / 1024.0;
        double total_gb = total_byte / 1024.0 / 1024.0 / 1024.0;
        double used_gb = total_gb - free_gb;

        result << "GPU Memory Info:\n";
        result << "  Total: " << std::fixed << std::setprecision(2) << total_gb << " GB\n";
        result << "  Used: " << used_gb << " GB\n";
        result << "  Free: " << free_gb << " GB\n";
        result << "  Usage: " << std::setprecision(1) << (used_gb / total_gb * 100.0) << "%";

        return result.str();
    }

    std::string CommandProcessor::handleTensorInfo(const std::string& tensor_name) {
        if (!scene_manager_) {
            return "No scene manager available";
        }

        if (tensor_name.empty()) {
            return "Usage: tensor_info <tensor_name>\nAvailable: means, scaling, rotation, shs, opacity";
        }

        const SplatData* model = scene_manager_->getModelForRendering();
        if (!model) {
            return "No model available";
        }

        torch::Tensor tensor;
        if (tensor_name == "means" || tensor_name == "positions") {
            tensor = model->get_means();
        } else if (tensor_name == "scales" || tensor_name == "scaling") {
            tensor = model->get_scaling();
        } else if (tensor_name == "rotations" || tensor_name == "rotation" || tensor_name == "quats") {
            tensor = model->get_rotation();
        } else if (tensor_name == "features" || tensor_name == "colors" || tensor_name == "shs") {
            tensor = model->get_shs();
        } else if (tensor_name == "opacities" || tensor_name == "opacity") {
            tensor = model->get_opacity();
        } else {
            return "Unknown tensor: " + tensor_name + "\nAvailable: means, scaling, rotation, shs, opacity";
        }

        std::ostringstream oss;
        oss << "Tensor '" << tensor_name << "' info:\n";
        oss << "  Shape: [";
        for (int i = 0; i < tensor.dim(); i++) {
            if (i > 0)
                oss << ", ";
            oss << tensor.size(i);
        }
        oss << "]\n";
        oss << "  Device: " << tensor.device() << "\n";
        oss << "  Dtype: " << tensor.dtype() << "\n";
        oss << "  Requires grad: " << (tensor.requires_grad() ? "Yes" : "No") << "\n";

        try {
            auto cpu_tensor = tensor.cpu();
            auto flat = cpu_tensor.flatten();
            if (flat.numel() > 0) {
                oss << "  Min: " << torch::min(flat).item<float>() << "\n";
                oss << "  Max: " << torch::max(flat).item<float>() << "\n";
                oss << "  Mean: " << torch::mean(flat).item<float>() << "\n";
                oss << "  Std: " << torch::std(flat).item<float>();
            }
        } catch (...) {
            oss << "  (Statistics unavailable)";
        }

        return oss.str();
    }

} // namespace gs