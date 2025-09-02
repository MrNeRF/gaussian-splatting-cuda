/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "rendering/rendering.hpp"
#include "rendering_engine_impl.hpp"

namespace gs::rendering {

    std::unique_ptr<RenderingEngine> RenderingEngine::create() {
        LOG_DEBUG("Creating RenderingEngine instance");
        return std::make_unique<RenderingEngineImpl>();
    }

} // namespace gs::rendering