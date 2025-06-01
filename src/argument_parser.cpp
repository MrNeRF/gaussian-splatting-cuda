// Copyright (c) 2023 Janusch Patas.

#include "core/argument_parser.hpp"
#include "core/parameters.hpp"
#include <args.hxx>
#include <filesystem>
#include <iostream>

namespace gs {
    namespace args {

        std::vector<std::string> convert_args(int argc, char* argv[]) {
            return std::vector<std::string>(argv, argv + argc);
        }

        int parse_arguments(const std::vector<std::string>& args,
                            gs::param::TrainingParameters& params) {

            if (args.empty()) {
                std::cerr << "No command line arguments provided!" << std::endl;
                return -1;
            }

            // Set up argument parser
            ::args::ArgumentParser parser(
                "3D Gaussian Splatting CUDA Implementation\n",
                "This program provides a lightning-fast CUDA implementation of the 3D Gaussian Splatting algorithm for real-time radiance field rendering.");

            // Define all command line arguments
            ::args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
            ::args::ValueFlag<float> convergence_rate(parser, "convergence_rate", "Set convergence rate", {'c', "convergence_rate"});
            ::args::ValueFlag<int> resolution(parser, "resolution", "Set resolution", {'r', "resolution"});
            ::args::Flag enable_cr_monitoring(parser, "enable_cr_monitoring", "Enable convergence rate monitoring", {"enable-cr-monitoring"});
            ::args::Flag force_overwrite_output_path(parser, "force", "Forces to overwrite output folder", {'f', "force"});
            ::args::Flag empty_gpu_memory(parser, "empty_gpu_cache", "Forces to reset GPU Cache. Should be lighter on VRAM", {"empty-gpu-cache"});
            ::args::ValueFlag<std::string> data_path(parser, "data_path", "Path to the training data", {'d', "data-path"});
            ::args::ValueFlag<std::string> output_path(parser, "output_path", "Path to the training output", {'o', "output-path"});
            ::args::ValueFlag<uint32_t> iterations(parser, "iterations", "Number of iterations to train the model", {'i', "iter"});
            ::args::CompletionFlag completion(parser, {"complete"});

            // Parse arguments
            try {
                parser.Prog(args.front());
                parser.ParseArgs(std::vector<std::string>(args.begin() + 1, args.end()));
            } catch (const ::args::Completion& e) {
                std::cout << e.what();
                return 0;
            } catch (const ::args::Help&) {
                std::cout << parser;
                return -1;
            } catch (const ::args::ParseError& e) {
                std::cerr << e.what() << std::endl;
                std::cerr << parser;
                return -1;
            }

            // Process parsed arguments and populate parameters

            // Data path is required
            if (::args::get(data_path).empty()) {
                std::cerr << "ERROR: No data path specified. Use --data_path to specify the path to the dataset.\n";
                return -1;
            }
            params.dataset.data_path = ::args::get(data_path);

            // Handle output path
            if (::args::get(output_path).empty()) {
                std::cerr << "ERROR: No output path specified. Use --output_path to specify the path for output files.\n";
                return -1;
            }
            params.dataset.output_path = ::args::get(output_path);

            // Process other arguments
            if (iterations) {
                params.optimization.iterations = ::args::get(iterations);
            }

            if (resolution) {
                params.dataset.resolution = ::args::get(resolution);
            }

            std::filesystem::path outputDir = params.dataset.output_path;
            if (!std::filesystem::exists(outputDir)) {
                std::filesystem::create_directories(outputDir);
            }
            params.dataset.output_path = outputDir;

            return 0; // Success
        }
    } // namespace args
} // namespace gs

gs::param::TrainingParameters gs::args::parse_args_and_params(int argc, char* argv[]) {
    gs::param::TrainingParameters params;
    params.optimization = gs::param::read_optim_params_from_json();
    if (parse_arguments(convert_args(argc, argv), params) < 0) {
        throw std::runtime_error("Failed to parse arguments");
    }
    return params;
}