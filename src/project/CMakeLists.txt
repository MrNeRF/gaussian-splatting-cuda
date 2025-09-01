# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Project management module
set(PROJECT_SOURCES
    project.cpp
)

add_library(gs_project STATIC ${PROJECT_SOURCES})

target_include_directories(gs_project
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(gs_project
    PUBLIC
        gs_core  # Depends on core
        nlohmann_json::nlohmann_json
)

set_target_properties(gs_project PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    DEBUG_POSTFIX d
)
