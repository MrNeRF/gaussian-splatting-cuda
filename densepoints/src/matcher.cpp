#include "matcher.h"

#include <vector>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> densepcd::match(torch::Tensor feats1, torch::Tensor feats2, float min_cossim) {
    auto cossim = torch::matmul(feats1, feats2.transpose(0, 1));
    auto [bestcossim, match12] = cossim.max(1);
    auto [bestcossim2, match21] = cossim.max(0);
    auto idx0 = torch::arange(match12.size(0), match12.device());
    auto indices12 = match21.index({match12});
    auto mask = indices12 == idx0;
    if (min_cossim > 0.f) {
        mask *= bestcossim > min_cossim;
    }
    return std::make_tuple(idx0, match12, mask);
}

densepcd::XFeatFeatureExtractor::XFeatFeatureExtractor(int image_width, int image_height, size_t max_keypoints, const std::string& model_directory)
 : _imageWidth(image_width), _imageHeight(image_height), _maxKeypoints(max_keypoints) {
    auto filename = std::format("xfeat_{}_{}_{}.pt", _imageWidth, _imageHeight, _maxKeypoints);
    auto path = DENSE_POINTS_DIR "/" + model_directory + "/" + filename;
    try {
        _network = torch::jit::load(path);
        _network.to(_device);
        _network.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model from: " << path.c_str() << std::endl;
        std::cerr << e.what() << std::endl;
    }
}

std::tuple<torch::Tensor, torch::Tensor> densepcd::XFeatFeatureExtractor::extract_features(torch::Tensor image) {
    torch::NoGradGuard guard;

    torch::Tensor inputImage = image.to(torch::kHalf);
    if (inputImage.dim() == 3) {
        inputImage = inputImage.unsqueeze(0);
    }
    float resizeFactors[2] = {1.f, 1.f};
    if (inputImage.size(2) != _imageHeight || inputImage.size(3) != _imageWidth) {
        float originalWidth = inputImage.size(3);
        float originalHeight = inputImage.size(2);
        auto inputImageResized = torch::nn::functional::interpolate(
            inputImage,
            torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({_imageHeight, _imageWidth})));
        inputImage = inputImageResized;
        resizeFactors[0] = originalWidth / (float)_imageWidth;
        resizeFactors[1] = originalHeight / (float)_imageHeight;
    }
    std::vector<torch::jit::IValue> input;
    input.push_back(inputImage);

    auto output = _network(input);
    auto out = output.toTuple()->elements();

    auto kpts = out[0].toTensor();
    auto desc = out[1].toTensor();

    auto resizeTensor = torch::from_blob(resizeFactors, {2}).to(torch::kCUDA);
    kpts = kpts * resizeTensor; // Move kpts back to original image space

    return std::make_tuple(kpts, desc);
}
