#pragma once

#include <torch/torch.h>
#include <torch/script.h>

#include <tuple>

namespace densepcd {
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> match(torch::Tensor feats1, torch::Tensor feats2, float min_cossim = 0.82f);

    class XFeatFeatureExtractor {
    private:
        torch::DeviceType _device = torch::kCUDA;
        torch::jit::Module _network;

        int _imageWidth;
        int _imageHeight;
        size_t _maxKeypoints;

    public:
        XFeatFeatureExtractor(int image_width, int image_height, size_t max_keypoints, const std::string& model_directory = "models");

        std::tuple<torch::Tensor, torch::Tensor> extract_features(torch::Tensor image);
    };

    torch::Tensor rgbToGray(torch::Tensor& input) {
        auto grayTensor = 0.299 * input[0] + 0.587 * input[1] + 0.114 * input[2];
        return grayTensor;
    }

    class LOFTRMatcher {
    private:
        torch::DeviceType _device = torch::kCUDA;
        torch::jit::Module _network;

        int _imageWidth = 640;
        int _imageHeight = 480;

        std::tuple<torch::Tensor, torch::Tensor> preprocessTensor(torch::Tensor tensor) {
            using namespace torch::indexing;

            torch::Tensor resizedTensor;
            float resizeFactors[2] = {1.f, 1.f};
            if (tensor.size(1) != _imageHeight || tensor.size(2) != _imageWidth) {
                float originalWidth = tensor.size(2);
                float originalHeight = tensor.size(1);
                resizedTensor = torch::nn::functional::interpolate(
                    tensor.unsqueeze(0),
                    torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({_imageHeight, _imageWidth})));
                resizedTensor = resizedTensor.squeeze(0);
                resizeFactors[0] = originalWidth / (float)_imageWidth;
                resizeFactors[1] = originalHeight / (float)_imageHeight;
            } else {
                // Image already correct size
                resizedTensor = tensor;
            }

            // Convert to gray if rgb
            if (resizedTensor.size(1) > 1) {
                resizedTensor = rgbToGray(resizedTensor).unsqueeze(0);
            }
            auto scaleTensor = torch::from_blob(resizeFactors, {2}).to(torch::kCUDA);
            return std::make_tuple(resizedTensor.unsqueeze(0), scaleTensor);
        }
    
    public:
        LOFTRMatcher(const std::string& model_directory = "models") {
            auto path = DENSE_POINTS_DIR "/" + model_directory + "/loftr.pth";
            try {
                _network = torch::jit::load(path);
                _network.to(_device);
                _network.eval();
            } catch (const c10::Error& e) {
                std::cerr << "Error loading the model from: " << path.c_str() << std::endl;
                std::cerr << e.what() << std::endl;
            }
        }

        std::tuple<torch::Tensor, torch::Tensor> matchImages(torch::Tensor image1, torch::Tensor image2, float confThreshold = -1.f) {
            torch::NoGradGuard guard;

            using namespace torch::indexing;

            torch::Tensor kpt1, kpt2;
            try {
                auto [input1, scale1] = preprocessTensor(image1);
                auto [input2, scale2] = preprocessTensor(image2);

                std::vector<torch::jit::IValue> input;
                input.push_back(input1);
                input.push_back(input2);

                auto output = _network(input);

                auto out = output.toTuple().get()[0];
                kpt1 = out.elements().at(0).toTensor();
                kpt2 = out.elements().at(1).toTensor();
                auto conf = out.elements().at(2).toTensor();

                if (confThreshold > 0) {
                    auto index = conf > confThreshold;
                    kpt1 = kpt1.index({index});
                    kpt2 = kpt2.index({index});
                }

                // Adj kpts to original size
                kpt1 = kpt1 * scale1;
                kpt2 = kpt2 * scale2;

            } catch (const c10::Error& e) {
                std::cerr << "Error running inference: " << e.what() << std::endl;
            }
            return std::make_tuple(kpt1, kpt2);
        }
    };
} // namespace densepcd