/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/utils/native_dialogs.hpp"
#include "core/logger.hpp"

#ifdef WIN32

namespace gs::gui::utils {

    HRESULT selectFileNative(PWSTR& strDirectory,
                             COMDLG_FILTERSPEC rgSpec[],
                             UINT cFileTypes,
                             bool blnDirectory) {

        HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
        if (FAILED(hr)) {
            LOG_ERROR("Failed to initialize COM: {:#x}", static_cast<unsigned int>(hr));
        } else {
            // Create the FileOpenDialog instance
            IFileOpenDialog* pFileOpen = nullptr;
            hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL,
                                  IID_IFileOpenDialog, reinterpret_cast<void**>(&pFileOpen));

            if (SUCCEEDED(hr)) {
                DWORD dwOptions;

                if (SUCCEEDED(pFileOpen->GetOptions(&dwOptions))) {
                    if (blnDirectory) {
                        pFileOpen->SetOptions(dwOptions | FOS_PICKFOLDERS);
                    } else {
                        if (rgSpec != nullptr && cFileTypes > 0) {
                            hr = pFileOpen->SetFileTypes(cFileTypes, rgSpec);
                            if (SUCCEEDED(hr)) {
                                pFileOpen->SetOptions(dwOptions | FOS_NOCHANGEDIR | FOS_FILEMUSTEXIST);
                                pFileOpen->SetFileTypeIndex(1);
                            } else {
                                LOG_ERROR("Failed to set file types: {:#x}", static_cast<unsigned int>(hr));
                            }
                        } else {
                            pFileOpen->SetOptions(dwOptions | FOS_NOCHANGEDIR | FOS_FILEMUSTEXIST);
                        }
                    }
                }

                // Show the Open File dialog
                hr = pFileOpen->Show(NULL);

                if (SUCCEEDED(hr)) {
                    IShellItem* pItem;
                    hr = pFileOpen->GetResult(&pItem);
                    if (SUCCEEDED(hr)) {
                        PWSTR filePath = nullptr;
                        hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &filePath);

                        if (SUCCEEDED(hr)) {
                            strDirectory = filePath;
                            CoTaskMemFree(filePath);
                        }
                        pItem->Release();
                    }
                }
                pFileOpen->Release();
            } else {
                LOG_ERROR("Failed to create FileOpenDialog: {:#x}", static_cast<unsigned int>(hr));
                CoUninitialize();
            }
            CoUninitialize();
        }
        return hr;
    }

} // namespace gs::gui::utils

#endif // WIN32
