#pragma once

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <unordered_map>

namespace test_utils {

    // Helper struct to hold test data
    struct TestData {
        torch::Tensor means;     // [N, 3]
        torch::Tensor quats;     // [N, 4]
        torch::Tensor scales;    // [N, 3]
        torch::Tensor opacities; // [N]
        torch::Tensor colors;    // [N, 3]
        torch::Tensor viewmats;  // [C, 4, 4]
        torch::Tensor Ks;        // [C, 3, 3]
        int width;
        int height;

        void to_device(torch::Device device) {
            means = means.to(device);
            quats = quats.to(device);
            scales = scales.to(device);
            opacities = opacities.to(device);
            colors = colors.to(device);
            viewmats = viewmats.to(device);
            Ks = Ks.to(device);
        }

        // Expand colors to match viewmats if needed
        void prepare_for_multi_view() {
            if (colors.dim() == 2 && viewmats.size(0) > 1) {
                colors = colors.unsqueeze(0).repeat({viewmats.size(0), 1, 1});
            }
        }
    };

    // Load test data from converted .pt file
    inline TestData load_test_data(const std::string& pt_path, torch::Device device) {
        TestData data;

        // Check if file exists
        std::ifstream check_file(pt_path);
        if (!check_file.good()) {
            throw std::runtime_error("Test data file not found: " + pt_path +
                                     "\nPlease ensure test_garden_data.pt exists in tests/data/");
        }
        check_file.close();

        try {
            // Load the pickled dictionary using torch::pickle
            std::vector<char> buffer;
            std::ifstream file(pt_path, std::ios::binary);
            file.seekg(0, std::ios::end);
            size_t file_size = file.tellg();
            buffer.resize(file_size);
            file.seekg(0, std::ios::beg);
            file.read(buffer.data(), file_size);
            file.close();

            // Unpickle the data
            auto unpickled = torch::pickle_load(buffer);
            auto dict = unpickled.toGenericDict();

            // Extract tensors
            data.means = dict.at("means").toTensor();
            data.quats = dict.at("quats").toTensor();
            data.scales = dict.at("scales").toTensor();
            data.opacities = dict.at("opacities").toTensor();
            data.colors = dict.at("colors").toTensor();
            data.viewmats = dict.at("viewmats").toTensor();
            data.Ks = dict.at("Ks").toTensor();

            // Extract scalars
            data.width = dict.at("width").toInt();
            data.height = dict.at("height").toInt();

            std::cout << "Loaded test data from: " << pt_path << std::endl;
            std::cout << "  Gaussians: " << data.means.size(0) << std::endl;
            std::cout << "  Cameras: " << data.viewmats.size(0) << std::endl;
            std::cout << "  Image size: " << data.width << "x" << data.height << std::endl;

        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to load test data: " + std::string(e.what()) +
                                     "\nMake sure the .pt file was saved with torch.save() in Python");
        }

        // Move to device
        data.to_device(device);

        // Prepare for multi-view if needed
        data.prepare_for_multi_view();

        return data;
    }

    // Alternative: Load with default path
    inline TestData load_test_data(torch::Device device) {
        // Try common locations
        std::vector<std::string> possible_paths = {
            "test_garden_data.pt",
            "tests/data/test_garden_data.pt",
            "../test_garden_data.pt",
            "../tests/data/test_garden_data.pt"};

        for (const auto& path : possible_paths) {
            std::ifstream check_file(path);
            if (check_file.good()) {
                check_file.close();
                return load_test_data(path, device);
            }
        }

        throw std::runtime_error(
            "Could not find test_garden_data.pt in any of the expected locations.\n"
            "Please ensure the file exists in tests/data/");
    }

} // namespace test_utils