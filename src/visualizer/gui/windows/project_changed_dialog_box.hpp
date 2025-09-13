/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
#include <functional>
#include <string>

namespace gs::gui {

    class ProjectChangedDialogBox {
    public:
        ProjectChangedDialogBox();

        void render(bool* p_open);
        void setOnDialogClose(std::function<void(bool)> callback);

    private:
        std::function<void(bool)> result_;
    };

} // namespace gs::gui
