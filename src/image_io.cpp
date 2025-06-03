#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "core/image_io.hpp"
#include "external/stb_image.h"
#include "external/stb_image_resize.h"
#include "external/stb_image_write.h"

#include <filesystem>
#include <vector>

// Synchronous image loading (existing implementation)
std::tuple<unsigned char*, int, int, int>
load_image(std::filesystem::path p, int res_div) {
    int w, h, c;
    unsigned char* img = stbi_load(p.string().c_str(), &w, &h, &c, 0);
    if (!img)
        throw std::runtime_error("Load failed: " + p.string() + " : " + stbi_failure_reason());

    if (res_div == 2 || res_div == 4 || res_div == 8) {
        int nw = w / res_div, nh = h / res_div;
        auto* out = static_cast<unsigned char*>(malloc(nw * nh * c));
        if (!stbir_resize_uint8(img, w, h, 0, out, nw, nh, 0, c))
            throw std::runtime_error("Resize failed: " + p.string() + " : " + stbi_failure_reason());
        stbi_image_free(img);
        img = out;
        w = nw;
        h = nh;
    }
    return {img, w, h, c};
}

void save_image(const std::filesystem::path& path, torch::Tensor image) {
    // Clone to avoid modifying original
    image = image.clone();

    // Ensure CPU and float
    image = image.to(torch::kCPU).to(torch::kFloat32);

    // Handle different input formats
    if (image.dim() == 4) { // [B, C, H, W] or [B, H, W, C]
        image = image.squeeze(0);
    }

    // Convert [C, H, W] to [H, W, C]
    if (image.dim() == 3 && image.size(0) <= 4) {
        image = image.permute({1, 2, 0});
    }

    // Make contiguous after permute
    image = image.contiguous();

    int height = image.size(0);
    int width = image.size(1);
    int channels = image.size(2);

    // Debug print
    std::cout << "Saving image: " << path << " shape: [" << height << ", " << width << ", " << channels << "]"
              << " min: " << image.min().item<float>()
              << " max: " << image.max().item<float>() << std::endl;

    // Convert to uint8
    auto img_uint8 = (image.clamp(0, 1) * 255).to(torch::kUInt8).contiguous();

    auto ext = path.extension().string();
    bool success = false;

    if (ext == ".png") {
        success = stbi_write_png(path.c_str(), width, height, channels,
                                 img_uint8.data_ptr<uint8_t>(), width * channels);
    } else if (ext == ".jpg" || ext == ".jpeg") {
        success = stbi_write_jpg(path.c_str(), width, height, channels,
                                 img_uint8.data_ptr<uint8_t>(), 95);
    }

    if (!success) {
        throw std::runtime_error("Failed to save image: " + path.string());
    }
}

void save_image(const std::filesystem::path& path,
                const std::vector<torch::Tensor>& images,
                bool horizontal,
                int separator_width) {
    if (images.empty()) {
        throw std::runtime_error("No images provided");
    }

    if (images.size() == 1) {
        save_image(path, images[0]);
        return;
    }

    // Prepare all images to same format
    std::vector<torch::Tensor> processed_images;
    for (auto img : images) {
        // Clone to avoid modifying original
        img = img.clone().to(torch::kCPU).to(torch::kFloat32);

        // Handle different input formats
        if (img.dim() == 4) {
            img = img.squeeze(0);
        }

        // Convert [C, H, W] to [H, W, C]
        if (img.dim() == 3 && img.size(0) <= 4) {
            img = img.permute({1, 2, 0});
        }

        processed_images.push_back(img.contiguous());
    }

    // Create separator (white by default)
    torch::Tensor separator;
    if (separator_width > 0) {
        auto first_img = processed_images[0];
        if (horizontal) {
            separator = torch::ones({first_img.size(0), separator_width, first_img.size(2)},
                                    first_img.options());
        } else {
            separator = torch::ones({separator_width, first_img.size(1), first_img.size(2)},
                                    first_img.options());
        }
    }

    // Concatenate images with separators
    torch::Tensor combined;
    for (size_t i = 0; i < processed_images.size(); ++i) {
        if (i == 0) {
            combined = processed_images[i];
        } else {
            if (separator_width > 0) {
                combined = torch::cat({combined, separator, processed_images[i]},
                                      horizontal ? 1 : 0);
            } else {
                combined = torch::cat({combined, processed_images[i]},
                                      horizontal ? 1 : 0);
            }
        }
    }

    // Save the combined image
    save_image(path, combined);
}

void free_image(unsigned char* img) {
    stbi_image_free(img);
}