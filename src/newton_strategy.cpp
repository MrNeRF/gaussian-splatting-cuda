#include "core/newton_strategy.hpp"
#include "core/rasterizer.hpp" // For gs::rasterize if needed for secondary targets
#include "core/torch_utils.hpp" // For get_bg_color_from_image

#include <algorithm> // for std::sort, std::nth_element
#include <limits>    // for std::numeric_limits

// Helper for spherical distance (assumes vectors are normalized relative to scene center)
static float spherical_distance(const Eigen::Vector3f& v1_on_sphere, const Eigen::Vector3f& v2_on_sphere) {
    float dot = v1_on_sphere.dot(v2_on_sphere); // Assumes v1, v2 are already normalized
    dot = std::max(-1.0f, std::min(1.0f, dot));
    return std::acos(dot);
}

NewtonStrategy::NewtonStrategy(
    const gs::param::TrainingParameters& training_params,
    SplatData& initial_splat_data, // TODO: Decide on ownership. For now, assuming it creates its own.
    std::shared_ptr<CameraDataset> train_dataset_for_knn)
: train_dataset_ref_(train_dataset_for_knn) {
    // The strategy should typically own its model.
    // It could be initialized from points, or by copying initial_splat_data if provided differently.
    // For now, let's assume it's initialized from TrainingParameters like in vanilla 3DGS.
    // This part needs to align with how Trainer sets up strategies.
    // If Trainer creates SplatData and passes it, then NewtonStrategy might take a reference or copy.
    // Let's assume for now the strategy creates its own SplatData based on training_params.
    // This often involves Colmap loading etc. which is complex.
    // A simpler start: clone initial_splat_data if it were passed as a const ref or by value.
    // For now, placeholder for proper SplatData initialization.
    // splat_data_ = std::make_unique<SplatData>(... copy from initial_splat_data or load ...);
    // This is a critical part that depends on the existing Trainer's design.
    // Let's assume SplatData is created and passed in, and we make a unique_ptr copy.
    // This constructor signature is likely to change based on Trainer's needs.

    // A more plausible scenario: Trainer creates the SplatData (model)
    // and passes it to the strategy constructor. The strategy then holds a reference or ptr.
    // For now, let's assume the strategy will create its own model based on parameters,
    // similar to how a typical training pipeline might start.
    // This will need to be adjusted. For the moment, let's make splat_data_ from scratch.
    // This is a temporary simplification.
    torch::Tensor scene_center_tensor = torch::zeros({3}, torch::kFloat32); // Placeholder
    splat_data_ = std::make_unique<SplatData>(SplatData::init_model_from_pointcloud(training_params, scene_center_tensor));

    // KNN data initialization
    if (train_dataset_ref_ && optim_params_cache_.use_newton_optimizer && optim_params_cache_.newton_knn_k > 0) {
        initialize_knn_data_if_needed();
    }
}

void NewtonStrategy::initialize(const gs::param::OptimizationParameters& optimParams) {
    optim_params_cache_ = optimParams; // Cache params

    if (optim_params_cache_.use_newton_optimizer) {
        NewtonOptimizer::Options newton_opts;
        newton_opts.step_scale = optim_params_cache_.newton_step_scale;
        newton_opts.damping = optim_params_cache_.newton_damping;
        newton_opts.knn_k = optim_params_cache_.newton_knn_k;
        newton_opts.secondary_target_downsample = optim_params_cache_.newton_secondary_target_downsample_factor;
        newton_opts.lambda_dssim_for_hessian = optim_params_cache_.newton_lambda_dssim_for_hessian;
        newton_opts.use_l2_for_hessian_L_term = optim_params_cache_.newton_use_l2_for_hessian_L_term;
        // Copy other relevant flags from optim_params_cache_ if NewtonOptimizer::Options has them

        optimizer_ = std::make_unique<NewtonOptimizer>(*splat_data_, optim_params_cache_, newton_opts);

        if (train_dataset_ref_ && newton_opts.knn_k > 0) {
            initialize_knn_data_if_needed();
        }
    } else {
        // Fallback or error if this strategy is used when use_newton_optimizer is false
        // Or, this strategy should only be created if use_newton_optimizer is true.
        TORCH_CHECK(false, "NewtonStrategy initialized but use_newton_optimizer is false in params!");
    }
}

void NewtonStrategy::compute_visibility_mask_for_model(const gs::RenderOutput& render_output, const SplatData& model) {
    // This is a placeholder. A robust solution needs the 'ranks' tensor from gsplat's projection output,
    // which maps the P_render Gaussians in render_output.visibility back to their original P_total indices.
    // render_output.visibility is a mask on P_render Gaussians.
    // We need a mask on P_total Gaussians.

    if (!render_output.visibility.defined() || model.size() == 0) {
        current_visibility_mask_for_model_ = torch::zeros({model.size()}, torch::kBool).to(model.get_means().device());
        return;
    }

    // Simplistic assumption: if render_output.visibility has P_total elements, use it directly.
    // This is unlikely to be correct.
    if (render_output.visibility.size(0) == model.size()) {
        current_visibility_mask_for_model_ = render_output.visibility.to(torch::kBool);
    } else {
        // Fallback: assume all are visible if mapping is unknown. This is not ideal for performance.
        // Or, assume none are if P_render is much smaller and no mapping.
        // For safety and to avoid processing non-rendered Gaussians, default to false if sizes mismatch badly.
        // A better placeholder: if render_output.visibility is for P_render, and we lack ranks,
        // we can't directly create a P_total mask from it.
        // The `visibility_mask_for_model` passed to NewtonOptimizer::step will be crucial.
        // For now, this strategy can't produce it correctly without `ranks`.
        // Let's set it to all false, relying on Trainer to perhaps provide a better one if possible,
        // or highlighting this as a deficiency.
        std::cerr << "Warning: Cannot accurately compute visibility_mask_for_model in NewtonStrategy without ranks tensor."
                  << "Using a conservative (all false) mask. Newton optimizer might not update much." << std::endl;
        current_visibility_mask_for_model_ = torch::zeros({model.size()}, torch::kBool).to(model.get_means().device());
        // A slightly better placeholder if render_output.visibility is boolean and for P_render:
        // Assume the first P_render elements of the model correspond to render_output.visibility
        // THIS IS A VERY STRONG AND LIKELY WRONG ASSUMPTION.
        // if (render_output.visibility.size(0) > 0 && render_output.visibility.size(0) <= model.size()) {
        //     current_visibility_mask_for_model_.slice(0, 0, render_output.visibility.size(0)) = render_output.visibility.to(torch::kBool);
        // }
    }
}


void NewtonStrategy::post_backward(int iter, gs::RenderOutput& render_output) {
    // This method is called by Trainer after loss.backward()
    // Cache necessary data for the NewtonOptimizer::step() call
    current_iter_ = iter;
    current_render_output_cache_ = render_output; // This is a shallow copy of Tensors if RenderOutput holds Tensors directly

    // Compute and cache the full visibility mask.
    // This is where the 'ranks' tensor from gsplat projection would be essential.
    // Since RenderOutput doesn't expose it, this will be a placeholder.
    compute_visibility_mask_for_model(render_output, *splat_data_);
}

void NewtonStrategy::step(int iter) {
    if (!optimizer_ || !optim_params_cache_.use_newton_optimizer) {
        TORCH_CHECK(false, "NewtonStrategy::step called without optimizer or when not enabled.");
        return;
    }
    if (!current_primary_camera_ || !current_primary_gt_image_.defined()) {
         TORCH_CHECK(false, "NewtonStrategy::step called without primary camera/GT image. Call set_current_view_data first.");
        return;
    }

    optimizer_->step(
        iter,
        current_visibility_mask_for_model_, // This needs to be correctly computed
        current_render_output_cache_,
        *current_primary_camera_,
        current_primary_gt_image_,
        current_knn_targets_gpu_ // KNN data (cameras and their GT images on GPU)
    );
}

bool NewtonStrategy::is_refining(int iter) const {
    // Basic refinement logic, can be adapted from standard 3DGS
    if (optim_params_cache_.refine_every > 0 && iter % optim_params_cache_.refine_every == 0) {
        return iter >= optim_params_cache_.start_refine && iter <= optim_params_cache_.stop_refine;
    }
    return false;
}

void NewtonStrategy::set_current_view_data(
    const Camera* primary_camera,
    const torch::Tensor& primary_gt_image,
    const gs::RenderOutput& render_output,
    const gs::param::OptimizationParameters& opt_params,
    int iteration
) {
    current_primary_camera_ = primary_camera;
    current_primary_gt_image_ = primary_gt_image;     // Assumed on device
    current_render_output_cache_ = render_output;     // Shallow copy of Tensors
    current_iter_ = iteration;
    // optim_params_cache_ should already be set by initialize()
    // Re-assert or update if dynamic changes are possible (unlikely for opt_params during training)
    // optim_params_cache_ = opt_params;

    compute_visibility_mask_for_model(render_output, *splat_data_);

    if (optim_params_cache_.use_newton_optimizer && optim_params_cache_.newton_knn_k > 0) {
        find_knn_for_current_primary(primary_camera);
    } else {
        current_knn_targets_gpu_.clear();
    }
}


void NewtonStrategy::initialize_knn_data_if_needed() {
    if (!train_dataset_ref_ || train_dataset_ref_->size().value_or(0) == 0) {
        std::cerr << "Warning: KNN data initialization skipped: no training dataset provided." << std::endl;
        return;
    }
    if (!all_train_cameras_cache_.empty() && !projected_camera_positions_on_sphere_.empty()) {
        return; // Already initialized
    }

    all_train_cameras_cache_.clear();
    projected_camera_positions_on_sphere_.clear();

    const auto& cameras_from_dataset = train_dataset_ref_->get_cameras_const();
    if (cameras_from_dataset.empty()) {
         std::cerr << "Warning: KNN data initialization skipped: training dataset has no cameras." << std::endl;
        return;
    }

    for (const auto& cam_wrapper : cameras_from_dataset) {
        all_train_cameras_cache_.push_back(cam_wrapper.get());
    }

    // Estimate scene center from camera positions (simplification)
    scene_center_for_knn_ = Eigen::Vector3f::Zero();
    for (const auto* cam : all_train_cameras_cache_) {
        // Assuming Camera class has a method to get Eigen::Vector3f center
        // If not, need to extract from its R, T or world_view_transform
        // Placeholder: cam->get_camera_center_eigen();
        // Let's compute from world_view_transform: C = -R^T t
        torch::Tensor V = cam->world_view_transform();
        if (V.defined()) {
            torch::Tensor R_wc = V.slice(0, 0, 3).slice(1, 0, 3).cpu(); // to CPU for Eigen
            torch::Tensor t_wc = V.slice(0, 0, 3).slice(1, 3, 4).cpu();
            Eigen::Matrix3f R_eigen;
            Eigen::Vector3f t_eigen;
            // Manual copy, TODO: find a more direct way if available via torch_utils for Eigen
            for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) R_eigen(r,c) = R_wc[r][c].item<float>();
            for(int r=0; r<3; ++r) t_eigen(r) = t_wc[r][0].item<float>();
            scene_center_for_knn_ += (-R_eigen.transpose() * t_eigen);
        }
    }
    if (!all_train_cameras_cache_.empty()) {
        scene_center_for_knn_ /= static_cast<float>(all_train_cameras_cache_.size());
    }

    scene_radius_for_knn_ = 0.f;
    for (const auto* cam : all_train_cameras_cache_) {
        torch::Tensor V = cam->world_view_transform();
        Eigen::Vector3f cam_center_world = Eigen::Vector3f::Zero();
         if (V.defined()) {
            torch::Tensor R_wc = V.slice(0, 0, 3).slice(1, 0, 3).cpu();
            torch::Tensor t_wc = V.slice(0, 0, 3).slice(1, 3, 4).cpu();
            Eigen::Matrix3f R_eigen; Eigen::Vector3f t_eigen;
            for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) R_eigen(r,c) = R_wc[r][c].item<float>();
            for(int r=0; r<3; ++r) t_eigen(r) = t_wc[r][0].item<float>();
            cam_center_world = (-R_eigen.transpose() * t_eigen);
        }
        scene_radius_for_knn_ = std::max(scene_radius_for_knn_, (cam_center_world - scene_center_for_knn_).norm());
    }
    if (scene_radius_for_knn_ < 1e-3f) scene_radius_for_knn_ = 1.0f;

    for (const auto* cam : all_train_cameras_cache_) {
        torch::Tensor V = cam->world_view_transform();
        Eigen::Vector3f cam_center_world = Eigen::Vector3f::Zero();
        if (V.defined()) {
            torch::Tensor R_wc = V.slice(0, 0, 3).slice(1, 0, 3).cpu();
            torch::Tensor t_wc = V.slice(0, 0, 3).slice(1, 3, 4).cpu();
            Eigen::Matrix3f R_eigen; Eigen::Vector3f t_eigen;
            for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) R_eigen(r,c) = R_wc[r][c].item<float>();
            for(int r=0; r<3; ++r) t_eigen(r) = t_wc[r][0].item<float>();
            cam_center_world = (-R_eigen.transpose() * t_eigen);
        }
        Eigen::Vector3f dir_from_scene_center = (cam_center_world - scene_center_for_knn_);
        if (dir_from_scene_center.norm() < 1e-6f) {
            projected_camera_positions_on_sphere_.push_back(Eigen::Vector3f(0,0,1));
        } else {
            projected_camera_positions_on_sphere_.push_back(dir_from_scene_center.normalized());
        }
    }
    std::cout << "KNN data initialized. Scene center: (" << scene_center_for_knn_.x() << ", "
              << scene_center_for_knn_.y() << ", " << scene_center_for_knn_.z()
              << "), Radius: " << scene_radius_for_knn_
              << ", Cameras processed: " << all_train_cameras_cache_.size() << std::endl;
}

void NewtonStrategy::find_knn_for_current_primary(const Camera* primary_cam_in) {
    current_knn_targets_gpu_.clear();
    if (optim_params_cache_.newton_knn_k == 0 || all_train_cameras_cache_.empty() || projected_camera_positions_on_sphere_.empty()) return;

    int primary_cam_idx = -1;
    for (size_t i = 0; i < all_train_cameras_cache_.size(); ++i) {
        if (all_train_cameras_cache_[i]->uid() == primary_cam_in->uid()) {
            primary_cam_idx = static_cast<int>(i);
            break;
        }
    }

    if (primary_cam_idx == -1) {
         std::cerr << "Warning: Primary camera for KNN not found in cached dataset." << std::endl;
        return;
    }

    const Eigen::Vector3f& primary_proj_pos = projected_camera_positions_on_sphere_[primary_cam_idx];
    std::vector<std::pair<float, int>> distances;
    for (size_t i = 0; i < all_train_cameras_cache_.size(); ++i) {
        if (static_cast<int>(i) == primary_cam_idx) continue;
        distances.emplace_back(spherical_distance(primary_proj_pos, projected_camera_positions_on_sphere_[i]), static_cast<int>(i));
    }

    if (distances.empty()) return;

    int actual_k = std::min(optim_params_cache_.newton_knn_k, static_cast<int>(distances.size()));
    std::nth_element(distances.begin(), distances.begin() + actual_k, distances.end());
    std::sort(distances.begin(), distances.begin() + actual_k);

    for (int i = 0; i < actual_k; ++i) {
        const Camera* secondary_cam = all_train_cameras_cache_[distances[i].second];
        // This assumes Camera has a method to load its image, potentially with resolution scaling.
        // The Camera class shown previously had `load_and_get_image(int resolution = -1)`.
        // We need to determine the target resolution for secondary GT images.
        int target_height = static_cast<int>(secondary_cam->image_height() * optim_params_cache_.newton_secondary_target_downsample_factor);
        int target_width = static_cast<int>(secondary_cam->image_width() * optim_params_cache_.newton_secondary_target_downsample_factor);

        // Create a temporary camera with new dimensions for loading if resolution param in load_and_get_image is not enough
        // Or, assume load_and_get_image handles downsampling if resolution is different from native.
        // For now, let's assume load_and_get_image can take a target resolution, or we post-process.
        // The Camera class provided doesn't show a way to change its internal H/W for loading.
        // So, we load full and then downsample.

        torch::Tensor secondary_gt_cpu = const_cast<Camera*>(secondary_cam)->load_and_get_image(); // Load full res CPU

        if (secondary_gt_cpu.defined() && secondary_gt_cpu.numel() > 0) {
             if (optim_params_cache_.newton_secondary_target_downsample_factor < 1.0f &&
                 optim_params_cache_.newton_secondary_target_downsample_factor > 0.0f) {

                // Ensure it's float and on CPU for interpolate if needed, then permute
                secondary_gt_cpu = secondary_gt_cpu.to(torch::kFloat32); // Ensure float for interpolate
                if (secondary_gt_cpu.is_cuda()) secondary_gt_cpu = secondary_gt_cpu.cpu();

                torch::Tensor input_for_interpolate = secondary_gt_cpu.permute({2,0,1}).unsqueeze(0); // HWC to 1CHW

                long new_H = static_cast<long>(secondary_gt_cpu.size(0) * optim_params_cache_.newton_secondary_target_downsample_factor);
                long new_W = static_cast<long>(secondary_gt_cpu.size(1) * optim_params_cache_.newton_secondary_target_downsample_factor);

                if (new_H > 0 && new_W > 0) {
                    secondary_gt_cpu = torch::nn::functional::interpolate(
                        input_for_interpolate,
                        torch::nn::functional::InterpolateFuncOptions().size(std::vector<long>{new_H, new_W}).mode(torch::kArea)
                    ).squeeze(0).permute({1,2,0}); // 1CHW -> CHW -> HWC
                } else {
                    std::cerr << "Warning: KNN downsampled GT image for cam " << secondary_cam->uid()
                              << " resulted in zero dimension. Original H/W: "
                              << secondary_gt_cpu.size(0) << "/" << secondary_gt_cpu.size(1)
                              << ", Factor: " << optim_params_cache_.newton_secondary_target_downsample_factor << std::endl;
                    continue; // Skip this problematic one
                }
            }
            current_knn_targets_gpu_.emplace_back(secondary_cam, secondary_gt_cpu.to(model_.get_means().device()));
        } else {
            std::cerr << "Warning: Could not load GT image for secondary KNN camera UID " << secondary_cam->uid() << std::endl;
        }
    }
}
