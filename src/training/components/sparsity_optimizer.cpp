#include "sparsity_optimizer.hpp"
#include "core/logger.hpp"
#include <format>
#include <print>

namespace gs::training {

    ADMMSparsityOptimizer::ADMMSparsityOptimizer(const Config& config)
        : config_(config) {
        LOG_DEBUG("Initializing ADMM sparsity optimizer with rho={}, prune_ratio={}, steps={}",
                  config_.init_rho, config_.prune_ratio, config_.sparsify_steps);
    }

    std::expected<void, std::string> ADMMSparsityOptimizer::initialize(const torch::Tensor& opacities) {
        try {
            torch::NoGradGuard no_grad;

            if (!opacities.defined() || opacities.numel() == 0) {
                return std::unexpected("Invalid opacity tensor for initialization");
            }

            // Initialize ADMM variables
            auto opa = torch::sigmoid(opacities).detach().contiguous();
            u_ = torch::zeros_like(opa);
            z_ = prune_z(opa + u_);
            initialized_ = true;

            LOG_DEBUG("ADMM sparsity optimizer initialized with {} Gaussians", opa.numel());
            return {};
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to initialize ADMM sparsity optimizer: {}", e.what());
            return std::unexpected(std::format("Failed to initialize ADMM optimizer: {}", e.what()));
        }
    }

    std::expected<torch::Tensor, std::string> ADMMSparsityOptimizer::compute_loss(const torch::Tensor& opacities) const {
        try {
            if (!initialized_) {
                LOG_WARN("ADMM optimizer compute_loss called before initialization");
                return torch::zeros({1}, torch::kFloat32).to(opacities.device()).requires_grad_();
            }

            if (!opacities.defined() || opacities.numel() == 0) {
                return std::unexpected("Invalid opacity tensor for loss computation");
            }

            // Compute ADMM sparsity loss
            auto opa = torch::sigmoid(opacities);
            auto diff = opa - z_.detach() + u_.detach();
            auto loss = 0.5f * config_.init_rho * torch::pow(torch::norm(diff, 2), 2);

            LOG_TRACE("ADMM sparsity loss: {}", loss.item<float>());
            return loss;
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to compute ADMM sparsity loss: {}", e.what());
            return std::unexpected(std::format("Failed to compute sparsity loss: {}", e.what()));
        }
    }

    std::expected<void, std::string> ADMMSparsityOptimizer::update_state(const torch::Tensor& opacities) {
        try {
            torch::NoGradGuard no_grad;

            if (!initialized_) {
                LOG_DEBUG("Initializing ADMM optimizer on first update");
                return initialize(opacities);
            }

            if (!opacities.defined() || opacities.numel() == 0) {
                return std::unexpected("Invalid opacity tensor for state update");
            }

            // ADMM update step
            auto opa = torch::sigmoid(opacities).detach().contiguous();
            auto z_temp = opa + u_;
            z_ = prune_z(z_temp);
            u_ += opa - z_;

            LOG_TRACE("ADMM state updated - ||u||={}, ||z||={}",
                      torch::norm(u_).item<float>(), torch::norm(z_).item<float>());
            return {};
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to update ADMM state: {}", e.what());
            return std::unexpected(std::format("Failed to update ADMM state: {}", e.what()));
        }
    }

    std::expected<torch::Tensor, std::string> ADMMSparsityOptimizer::get_prune_mask(const torch::Tensor& opacities) const {
        try {
            if (!opacities.defined() || opacities.numel() == 0) {
                return std::unexpected("Invalid opacity tensor for pruning");
            }

            auto opa = torch::sigmoid(opacities.flatten());
            int n_prune = static_cast<int>(config_.prune_ratio * opa.size(0));

            if (n_prune == 0) {
                LOG_DEBUG("No Gaussians to prune (n_prune=0)");
                return torch::zeros(opa.size(0), torch::kBool).to(opa.device());
            }

            // Find indices of smallest opacities
            auto [_, prune_indices] = torch::topk(opa, n_prune, -1, /*largest=*/false);

            // Create boolean mask
            auto mask = torch::zeros(opa.size(0), torch::kBool).to(opa.device());
            mask.index_put_({prune_indices}, true);

            LOG_DEBUG("Generated prune mask for {} out of {} Gaussians",
                      n_prune, opa.size(0));
            return mask;
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to generate prune mask: {}", e.what());
            return std::unexpected(std::format("Failed to generate prune mask: {}", e.what()));
        }
    }

    int ADMMSparsityOptimizer::get_num_to_prune(const torch::Tensor& opacities) const {
        if (!opacities.defined() || opacities.numel() == 0) {
            return 0;
        }
        return static_cast<int>(config_.prune_ratio * opacities.flatten().size(0));
    }

    torch::Tensor ADMMSparsityOptimizer::prune_z(const torch::Tensor& z) const {
        if (z.numel() == 0) {
            return torch::zeros_like(z);
        }

        int index = static_cast<int>(config_.prune_ratio * z.size(0));
        if (index == 0) {
            return torch::zeros_like(z);
        }

        // Sort to find threshold
        auto [z_sorted, _] = torch::sort(z.flatten(), 0);
        auto z_threshold = z_sorted[index - 1];

        // Apply soft thresholding
        return (z > z_threshold) * z;
    }

    // Factory implementation
    std::unique_ptr<ISparsityOptimizer> SparsityOptimizerFactory::create(
        const std::string& method,
        const ADMMSparsityOptimizer::Config& config) {
        if (method == "admm") {
            return std::make_unique<ADMMSparsityOptimizer>(config);
        }
        LOG_ERROR("Unknown sparsity optimization method: {}", method);
        return nullptr;
    }

} // namespace gs::training
