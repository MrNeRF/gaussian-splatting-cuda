/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#ifdef WIN32
#include <Shobjidl.h>
#include <Windows.h>
#include <filesystem>
#endif

namespace gs::gui::utils {

#ifdef WIN32
    /**
     * Opens a native Windows file/folder selection dialog
     * @param strDirectory Output path selected by the user
     * @param rgSpec File type filters (can be nullptr)
     * @param cFileTypes Number of file type filters
     * @param blnDirectory True to select folders, false for files
     * @return HRESULT indicating success or failure
     */
    HRESULT selectFileNative(PWSTR& strDirectory,
                             COMDLG_FILTERSPEC rgSpec[] = nullptr,
                             UINT cFileTypes = 0,
                             bool blnDirectory = false);
#endif

} // namespace gs::gui::utils
