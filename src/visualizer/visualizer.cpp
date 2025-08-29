/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/visualizer.hpp"
#include "visualizer_impl.hpp"

namespace gs::visualizer {

    std::unique_ptr<Visualizer> Visualizer::create(const ViewerOptions& options) {
        return std::make_unique<VisualizerImpl>(options);
    }

} // namespace gs::visualizer
