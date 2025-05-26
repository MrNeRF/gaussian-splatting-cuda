// Copyright (c) 2023 Janusch Patas.

#include "core/argument_parser.hpp"
#include "core/parameters.hpp"
#include <args.hxx>
#include <filesystem>
#include <iostream>

namespace gs {
    namespace args {

        std::vector<std::string> convert_args(int argc, char* argv[]) {
            std::vector<std::string> args;
            args.reserve(argc);

            for (int i = 0; i < argc; ++i) {
                args.emplace_back(argv[i]);
            }

            return args;
        }

        int parse_arguments(const std::vector<std::string>& args,
                            gs::param::ModelParameters& modelParams,
                            gs::param::OptimizationParameters& optimParams) {

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
            if (data_path) {
                modelParams.source_path = ::args::get(data_path);
            } else {
                std::cerr << "No data path provided!" << std::endl;
                return -1;
            }

            // Handle output path
            if (output_path) {
                modelParams.output_path = ::args::get(output_path);
            } else {
                // Create default output path
                std::filesystem::path executablePath = std::filesystem::canonical("/proc/self/exe");
                std::filesystem::path parentDir = executablePath.parent_path().parent_path();
                std::filesystem::path outputDir = parentDir / "output";

                try {
                    bool isCreated = std::filesystem::create_directory(outputDir);
                    if (!isCreated) {
                        if (!force_overwrite_output_path) {
                            std::cerr << "Directory already exists! Not overwriting it" << std::endl;
                            return -1;
                        } else {
                            std::filesystem::create_directory(outputDir);
                            std::filesystem::remove_all(outputDir);
                        }
                    }
                } catch (...) {
                    std::cerr << "Failed to create output directory!" << std::endl;
                    return -1;
                }
                modelParams.output_path = outputDir;
            }

            // Process other arguments
            if (iterations) {
                optimParams.iterations = ::args::get(iterations);
            }

            if (resolution) {
                modelParams.resolution = ::args::get(resolution);
            }

            // GPU cache setting
            optimParams.empty_gpu_cache = ::args::get(empty_gpu_memory);

            return 0; // Success
        }
    } // namespace args
} // namespace gs