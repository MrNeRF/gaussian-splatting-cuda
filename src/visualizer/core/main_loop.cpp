/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "main_loop.hpp"
#include "core/logger.hpp"

namespace gs::visualizer {

    void MainLoop::run() {
        LOG_INFO("Main loop starting");

        if (init_callback_) {
            if (!init_callback_()) {
                LOG_ERROR("Initialization failed");
                return;
            }
        }

        LOG_DEBUG("Entering main render loop");

        // Continue running while:
        // - Either we don't have a should_close callback (run forever)
        // - OR we have one and it returns false (not time to close)
        while (true) {
            if (!should_close_callback_ || should_close_callback_()) {
                LOG_DEBUG("Should close callback requested exit");
                break;
            }

            if (update_callback_) {
                update_callback_();
            }

            if (render_callback_) {
                render_callback_();
            }
        }

        LOG_DEBUG("Exiting main render loop");

        if (shutdown_callback_) {
            shutdown_callback_();
        }

        LOG_INFO("Main loop ended");
    }

} // namespace gs::visualizer