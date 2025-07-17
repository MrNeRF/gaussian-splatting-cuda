#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#ifdef VCPKG_PYTHON_AVAILABLE
#ifndef PYTHON_EXECUTABLE
#error "PYTHON_EXECUTABLE not defined"
#endif
#endif

class VcpkgPythonDownloader {
private:
    std::string python_executable;

public:
    VcpkgPythonDownloader() {
#ifdef VCPKG_PYTHON_AVAILABLE
        python_executable = PYTHON_EXECUTABLE;
#else
        // Fallback to system Python
        python_executable = "python3";
#endif
    }

    bool verify_python() {
        std::string test_cmd = python_executable + " --version";

        std::cout << "Testing Python: " << python_executable << std::endl;

        int result = system(test_cmd.c_str());
        if (result == 0) {
            std::cout << "✓ Python is working!" << std::endl;
            return true;
        } else {
            std::cerr << "✗ Python test failed!" << std::endl;
            return false;
        }
    }

    bool run_download_script(const std::vector<std::string>& args) {
        if (!verify_python()) {
            return false;
        }

        // Check if script exists
        if (!std::filesystem::exists("scripts/download_dataset.py")) {
            std::cerr << "Error: scripts/download_dataset.py not found!" << std::endl;
            std::cerr << "Make sure you're running from the project root directory." << std::endl;
            return false;
        }

        // Build command
        std::string command = python_executable + " scripts/download_dataset.py";
        for (const auto& arg : args) {
            command += " \"" + arg + "\""; // Quote arguments to handle spaces
        }

        std::cout << "Executing: " << command << std::endl;

        int result = system(command.c_str());
        return result == 0;
    }

    void show_info() {
        std::cout << "Dataset Downloader (vcpkg Python3 edition)" << std::endl;
        std::cout << "Python executable: " << python_executable << std::endl;

#ifdef VCPKG_PYTHON_AVAILABLE
        std::cout << "Using vcpkg-managed Python" << std::endl;
#else
        std::cout << "Using system Python fallback" << std::endl;
#endif
    }
};

int main(int argc, char* argv[]) {
    VcpkgPythonDownloader downloader;

    if (argc < 2) {
        downloader.show_info();
        std::cout << "\nUsage: downloader <dataset_name> [--save-dir <path>]" << std::endl;
        std::cout << "Available datasets: mipnerf360, mipnerf360_extra, bilarf, zipnerf, zipnerf_undistorted" << std::endl;
        return 1;
    }

    // Prepare arguments for Python script
    std::vector<std::string> script_args;
    for (int i = 1; i < argc; i++) {
        script_args.push_back(argv[i]);
    }

    // Run the download script
    if (downloader.run_download_script(script_args)) {
        std::cout << "✓ Download completed successfully!" << std::endl;
        return 0;
    } else {
        std::cerr << "✗ Download failed!" << std::endl;
        return 1;
    }
}
