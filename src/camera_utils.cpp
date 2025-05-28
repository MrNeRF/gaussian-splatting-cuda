#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include "core/camera_utils.hpp"
#include "external/stb_image.h"
#include "external/stb_image_resize.h"

#include <cmath>
#include <filesystem>
#include <iostream>

// -----------------------------------------------------------------------------
//  Image I/O helpers
// -----------------------------------------------------------------------------
std::tuple<unsigned char*, int, int, int>
read_image(std::filesystem::path p, int res_div) {
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

void free_image(unsigned char* img) { stbi_image_free(img); }
