//
// Created by paja on 8/30/24.
//

#include "args_parser.cuh"
#include <args.hxx>
#include <string>
#include <vector>
#include "parameters.cuh"
#include <filesystem>
#include <fstream>

int ArgsParser::Parse(const int argc, const char* argv[],
                        gs::param::ModelParameters& modelParams,
                        gs::param::OptimizationParameters& optimParams) {

    if (argc <= 1) {
        std::cerr << "No command line arguments provided!" << std::endl;
        return -1;
    }

    auto args = std::vector<std::string>();
    args.reserve(argc);

    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }
    args::ArgumentParser parser("3D Gaussian Splatting CUDA Implementation\n",
                                "This program provides a lightning-fast CUDA implementation of the 3D Gaussian Splatting algorithm for real-time radiance field rendering.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<float> convergence_rate(parser, "convergence_rate", "Set convergence rate", {'c', "convergence_rate"});
    args::ValueFlag<int> resolution(parser, "resolution", "Set resolutino", {'r', "resolution"});
    args::Flag enable_cr_monitoring(parser, "enable_cr_monitoring", "Enable convergence rate monitoring", {"enable-cr-monitoring"});
    args::Flag force_overwrite_output_path(parser, "force", "Forces to overwrite output folder", {'f', "force"});
    args::Flag empty_gpu_memory(parser, "empty_gpu_cache", "Forces to reset GPU Cache. Should be lighter on VRAM", {"empty-gpu-cache"});
    args::ValueFlag<std::string> data_path(parser, "data_path", "Path to the training data", {'d', "data-path"});
    args::ValueFlag<std::string> output_path(parser, "output_path", "Path to the training output", {'o', "output-path"});
    args::ValueFlag<uint32_t> iterations(parser, "iterations", "Number of iterations to train the model", {'i', "iter"});
    args::CompletionFlag completion(parser, {"complete"});

    try {
        parser.Prog(args.front());
        parser.ParseArgs(std::vector<std::string>(args.begin() + 1, args.end()));
    } catch (const args::Completion& e) {
        std::cout << e.what();
        return 0;
    } catch (const args::Help&) {
        std::cout << parser;
        return -1;
    } catch (const args::ParseError& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return -1;
    }

    if (data_path) {
        modelParams.source_path = args::get(data_path);
    } else {
        std::cerr << "No data path provided!" << std::endl;
        return -1;
    }
    if (output_path) {
        modelParams.output_path = args::get(output_path);
    } else {
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

    if (iterations) {
        optimParams.iterations = args::get(iterations);
    }

    if (resolution) {
        modelParams.resolution = args::get(resolution);
    }

    optimParams.empty_gpu_cache = args::get(empty_gpu_memory);
    return 0;
}

int ArgsParser::Dump(const gs::param::ModelParameters& params) {
    std::filesystem::path outputPath = params.output_path;
    std::filesystem::create_directories(outputPath); // Make sure the directory exists

    std::ofstream cfg_log_f(outputPath / "cfg_args");
    if (!cfg_log_f.is_open()) {
        std::cerr << "Failed to open file for writing!" << std::endl;
        return -1;
    }

    // Write the parameters in the desired format
    cfg_log_f << "Namespace(";
    cfg_log_f << "eval=" << (params.eval ? "True" : "False") << ", ";
    cfg_log_f << "images='" << params.images << "', ";
    cfg_log_f << "model_path='" << params.output_path.string() << "', ";
    cfg_log_f << "resolution=" << params.resolution << ", ";
    cfg_log_f << "sh_degree=" << params.sh_degree << ", ";
    cfg_log_f << "source_path='" << params.source_path.string() << "', ";
    cfg_log_f << "white_background=" << (params.white_background ? "True" : "False") << ")";
    cfg_log_f.close();

    std::cout << "Output folder: " << params.output_path.string() << std::endl;

    return 0;
}

