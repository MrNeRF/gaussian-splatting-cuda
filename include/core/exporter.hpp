#pragma once
#include "external/tinyply.hpp"
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <torch/torch.h>
#include <vector>

/* ------------------------------------------------------------------------- */
/*  PLAIN-OLD DATA SNAPSHOT                                                  */
/* ------------------------------------------------------------------------- */
struct GaussianPointCloud {
    torch::Tensor xyz, normals,
        features_dc, features_rest,
        opacity, scaling, rotation;
    std::vector<std::string> attribute_names;
};

/* ------------------------------------------------------------------------- */
/*  ATTRIBUTE-LIST BUILDER                                                   */
/* ------------------------------------------------------------------------- */
inline std::vector<std::string>
make_attribute_names(const torch::Tensor& f_dc,
                     const torch::Tensor& f_rest,
                     const torch::Tensor& scaling,
                     const torch::Tensor& rotation) {
    std::vector<std::string> a{"x", "y", "z", "nx", "ny", "nz"};

    for (int i = 0; i < f_dc.size(1) * f_dc.size(2); ++i)
        a.emplace_back("f_dc_" + std::to_string(i));
    for (int i = 0; i < f_rest.size(1) * f_rest.size(2); ++i)
        a.emplace_back("f_rest_" + std::to_string(i));

    a.emplace_back("opacity");

    for (int i = 0; i < scaling.size(1); ++i)
        a.emplace_back("scale_" + std::to_string(i));
    for (int i = 0; i < rotation.size(1); ++i)
        a.emplace_back("rot_" + std::to_string(i));
    return a;
}

/* ------------------------------------------------------------------------- */
/*  EXPORTER                                                                 */
/* ------------------------------------------------------------------------- */
inline void write_ply(const GaussianPointCloud& pc,
                      const std::filesystem::path& root,
                      int iteration,
                      bool join_thread = false) {
    namespace fs = std::filesystem;
    fs::path folder = root / ("point_cloud/iteration_" + std::to_string(iteration));
    fs::create_directories(folder);

    /* ----- pack all per-vertex tensors in the order we want to write ----- */
    std::vector<torch::Tensor> tensors{
        pc.xyz, pc.normals, pc.features_dc, pc.features_rest,
        pc.opacity, pc.scaling, pc.rotation};

    /* ----- background job ------------------------------------------------ */
    std::thread t([folder,
                   tensors = std::move(tensors),
                   names = pc.attribute_names]() mutable {
        /* ---- local lambda that owns the actual tinyply call ------------- */
        auto write_output_ply =
            [](const fs::path& file_path,
               const std::vector<torch::Tensor>& data,
               const std::vector<std::string>& attr_names) {
                tinyply::PlyFile ply;
                size_t attr_off = 0;

                for (const auto& tensor : data) {
                    const size_t cols = tensor.size(1);
                    std::vector<std::string> attrs(attr_names.begin() + attr_off,
                                                   attr_names.begin() + attr_off + cols);

                    ply.add_properties_to_element(
                        "vertex",
                        attrs,
                        tinyply::Type::FLOAT32,
                        tensor.size(0),
                        reinterpret_cast<uint8_t*>(tensor.data_ptr<float>()),
                        tinyply::Type::INVALID, 0);

                    attr_off += cols;
                }

                std::filebuf fb;
                fb.open(file_path, std::ios::out | std::ios::binary);
                std::ostream out_stream(&fb);
                ply.write(out_stream, /*binary=*/true);
            };

        write_output_ply(folder / "point_cloud.ply", tensors, names);
    });

    join_thread ? t.join()
                : t.detach();
}
