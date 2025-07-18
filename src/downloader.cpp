#include <algorithm>
#include <array>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <iostream>
#include <print>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifdef VCPKG_PYTHON_AVAILABLE
#ifndef PYTHON_EXECUTABLE
#error "PYTHON_EXECUTABLE not defined"
#endif
#endif

using namespace std::string_view_literals;
namespace fs = std::filesystem;

class VcpkgPythonDownloader {
private:
    std::string python_executable_;
    static constexpr auto SCRIPT_PATH = "scripts/download_dataset.py"sv;
    static constexpr auto AVAILABLE_DATASETS = std::array{
        "mipnerf360"sv, "mipnerf360_extra"sv, "bilarf"sv,
        "zipnerf"sv, "zipnerf_undistorted"sv};

    // Validate that the python executable path is safe
    [[nodiscard]] static auto is_safe_executable_path(std::string_view path) noexcept -> bool {
        // Basic validation - no shell metacharacters
        constexpr auto dangerous_chars = ";|&`$(){}[]<>*?~"sv;
        return !std::ranges::any_of(dangerous_chars, [path](char c) {
            return path.contains(c);
        });
    }

    // Validate dataset name against known safe values
    [[nodiscard]] static auto is_valid_dataset(std::string_view dataset) noexcept -> bool {
        return std::ranges::any_of(AVAILABLE_DATASETS, [dataset](auto valid) {
            return valid == dataset;
        });
    }

    // Secure process execution without shell interpretation
    [[nodiscard]] auto execute_process(const std::vector<std::string>& args) const -> std::expected<int, std::string> {
#ifdef _WIN32
        // Windows implementation using CreateProcess
        std::string command_line = python_executable_;
        for (const auto& arg : args) {
            command_line += " \"" + arg + "\"";
        }

        STARTUPINFOA si{};
        PROCESS_INFORMATION pi{};
        si.cb = sizeof(si);

        if (!CreateProcessA(
                nullptr,
                command_line.data(),
                nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
            return std::unexpected("Failed to create process");
        }

        WaitForSingleObject(pi.hProcess, INFINITE);

        DWORD exit_code;
        GetExitCodeProcess(pi.hProcess, &exit_code);

        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);

        return static_cast<int>(exit_code);
#else
        // Unix implementation using fork/exec
        pid_t pid = fork();
        if (pid == -1) {
            return std::unexpected("Failed to fork process");
        }

        if (pid == 0) {
            // Child process
            std::vector<char*> argv_vec;
            argv_vec.reserve(args.size() + 2);

            // Add python executable
            argv_vec.push_back(const_cast<char*>(python_executable_.c_str()));

            // Add arguments
            for (const auto& arg : args) {
                argv_vec.push_back(const_cast<char*>(arg.c_str()));
            }
            argv_vec.push_back(nullptr);

            execvp(python_executable_.c_str(), argv_vec.data());
            _exit(127); // exec failed
        } else {
            // Parent process
            int status;
            if (waitpid(pid, &status, 0) == -1) {
                return std::unexpected("Failed to wait for child process");
            }

            if (WIFEXITED(status)) {
                return WEXITSTATUS(status);
            } else {
                return std::unexpected("Process terminated abnormally");
            }
        }
#endif
    }

public:
    VcpkgPythonDownloader() {
#ifdef VCPKG_PYTHON_AVAILABLE
        python_executable_ = PYTHON_EXECUTABLE;
#else
        python_executable_ = "python3";
#endif
    }

    [[nodiscard]] auto verify_python() const -> std::expected<void, std::string> {
        if (!is_safe_executable_path(python_executable_)) {
            return std::unexpected("Python executable path contains unsafe characters");
        }

        std::println("Testing Python: {}", python_executable_);

        auto result = execute_process({"--version"});
        if (!result) {
            return std::unexpected(std::format("Python test failed: {}", result.error()));
        }

        if (result.value() == 0) {
            std::println("✓ Python is working!");
            return {};
        } else {
            return std::unexpected(std::format("Python test failed with exit code: {}", result.value()));
        }
    }

    [[nodiscard]] auto run_download_script(std::span<const std::string_view> args) const
        -> std::expected<void, std::string> {

        if (auto result = verify_python(); !result) {
            return result;
        }

        // Check if script exists
        if (!fs::exists(SCRIPT_PATH)) {
            return std::unexpected(std::format(
                "Error: {} not found!\n"
                "Make sure you're running from the project root directory.",
                SCRIPT_PATH));
        }

        // Check for special flags that don't require dataset validation
        bool has_list_flag = std::ranges::any_of(args, [](std::string_view arg) {
            return arg == "--list";
        });

        if (!has_list_flag) {
            // Only validate dataset if not using --list
            if (args.empty()) {
                return std::unexpected("No dataset specified");
            }

            const auto dataset = args[0];
            if (!is_valid_dataset(dataset)) {
                auto available = AVAILABLE_DATASETS | std::ranges::views::transform([](auto sv) { return std::string{sv}; }) | std::ranges::to<std::vector>();
                std::string datasets_str;
                for (size_t i = 0; i < available.size(); ++i) {
                    if (i > 0)
                        datasets_str += ", ";
                    datasets_str += available[i];
                }

                return std::unexpected(std::format(
                    "Invalid dataset '{}'. Available datasets: {}",
                    dataset, datasets_str));
            }
        }

        // Build secure argument vector
        std::vector<std::string> process_args;
        process_args.reserve(args.size() + 1);
        process_args.emplace_back(SCRIPT_PATH);

        for (const auto& arg : args) {
            // Allow common flags to pass through without validation
            if (arg.starts_with("--")) {
                // Skip validation for flags like --list, --save-dir, etc.
            } else {
                // Basic validation - reject arguments with shell metacharacters
                if (arg.contains(';') || arg.contains('|') || arg.contains('&') ||
                    arg.contains('`') || arg.contains('$')) {
                    return std::unexpected(std::format("Unsafe argument detected: {}", arg));
                }
            }
            process_args.emplace_back(arg);
        }

        // Display command for transparency
        std::print("Executing: {} ", python_executable_);
        for (size_t i = 0; i < process_args.size(); ++i) {
            if (i > 0)
                std::print(" ");
            std::print("{}", process_args[i]);
        }
        std::println();

        auto result = execute_process(process_args);
        if (!result) {
            return std::unexpected(std::format("Script execution failed: {}", result.error()));
        }

        if (result.value() == 0) {
            return {};
        } else {
            return std::unexpected(std::format("Script execution failed with exit code: {}", result.value()));
        }
    }

    auto show_info() const -> void {
        std::println("Dataset Downloader (vcpkg Python3 edition)");
        std::println("Python executable: {}", python_executable_);

#ifdef VCPKG_PYTHON_AVAILABLE
        std::println("Using vcpkg-managed Python");
#else
        std::println("Using system Python fallback");
#endif

        std::println("\nAvailable datasets:");
        for (const auto& dataset : AVAILABLE_DATASETS) {
            std::println("  - {}", dataset);
        }
    }
};

auto main(int argc, char* argv[]) -> int {
    const VcpkgPythonDownloader downloader;

    if (argc < 2) {
        downloader.show_info();
        std::println("\nUsage: {} <dataset_name> [--save-dir <path>]", argv[0]);
        return 1;
    }

    // Convert C-style args to modern span of string_views
    const auto args = std::span{argv + 1, static_cast<std::size_t>(argc - 1)} | std::ranges::views::transform([](const char* arg) { return std::string_view{arg}; }) | std::ranges::to<std::vector<std::string_view>>();

    // Run the download script
    if (auto result = downloader.run_download_script(args); result) {
        std::println("✓ Download completed successfully!");
        return 0;
    } else {
        std::println(stderr, "✗ Download failed: {}", result.error());
        return 1;
    }
}