#include "core/read_utils.hpp"
#include "core/camera_info.hpp"
#include "core/camera_utils.hpp"
#include "core/image.hpp"
#include "core/point_cloud.hpp"

#include <Eigen/Core>
#include <torch/torch.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <tinyply.h>
#include <unordered_map>
#include <vector>

// -----------------------------------------------------------------------------
//  Small POD read helpers (zero‑overhead after inlining)
// -----------------------------------------------------------------------------
static inline uint64_t read_u64(const char*& p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    p += 8;
    return v;
}
static inline uint32_t read_u32(const char*& p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    p += 4;
    return v;
}
static inline int32_t read_i32(const char*& p) {
    int32_t v;
    std::memcpy(&v, p, 4);
    p += 4;
    return v;
}
static inline double read_f64(const char*& p) {
    double v;
    std::memcpy(&v, p, 8);
    p += 8;
    return v;
}

// -----------------------------------------------------------------------------
//  Mapping of COLMAP camera model IDs → (enum, parameter‑count)
// -----------------------------------------------------------------------------
std::unordered_map<int, std::pair<CAMERA_MODEL, int32_t>> camera_model_ids = {
    {0, {CAMERA_MODEL::SIMPLE_PINHOLE, 3}},
    {1, {CAMERA_MODEL::PINHOLE, 4}},
    {2, {CAMERA_MODEL::SIMPLE_RADIAL, 4}},
    {3, {CAMERA_MODEL::RADIAL, 5}},
    {4, {CAMERA_MODEL::OPENCV, 8}},
    {5, {CAMERA_MODEL::OPENCV_FISHEYE, 8}},
    {6, {CAMERA_MODEL::FULL_OPENCV, 12}},
    {7, {CAMERA_MODEL::FOV, 5}},
    {8, {CAMERA_MODEL::SIMPLE_RADIAL_FISHEYE, 4}},
    {9, {CAMERA_MODEL::RADIAL_FISHEYE, 5}},
    {10, {CAMERA_MODEL::THIN_PRISM_FISHEYE, 12}},
    {11, {CAMERA_MODEL::UNDEFINED, -1}}};

// -----------------------------------------------------------------------------
//  File helpers
// -----------------------------------------------------------------------------
std::unique_ptr<std::vector<char>> read_binary(const std::filesystem::path& file_path) {
    std::ifstream f(file_path, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error("Failed to open " + file_path.string());

    const auto size = static_cast<std::streamsize>(f.tellg());
    auto buffer = std::make_unique<std::vector<char>>(static_cast<size_t>(size));

    f.seekg(0, std::ios::beg);
    f.read(buffer->data(), size);
    if (!f)
        throw std::runtime_error("Short read on " + file_path.string());

    return buffer; // one allocation, one read, zero extra copies
}

// -----------------------------------------------------------------------------
//  PLY writer for debugging / export
// -----------------------------------------------------------------------------
void Write_output_ply(const std::filesystem::path& file_path,
                      const std::vector<torch::Tensor>& tensors,
                      const std::vector<std::string>& attribute_names) {
    tinyply::PlyFile ply_file;

    size_t attr_offset = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const size_t cols = tensors[i].size(1);
        std::vector<std::string> attrs;
        attrs.reserve(cols);
        for (size_t j = 0; j < cols; ++j)
            attrs.push_back(attribute_names[attr_offset + j]);

        ply_file.add_properties_to_element(
            "vertex", attrs, tinyply::Type::FLOAT32, tensors[i].size(0),
            reinterpret_cast<uint8_t*>(tensors[i].data_ptr<float>()),
            tinyply::Type::INVALID, 0);

        attr_offset += cols;
    }

    std::filebuf fb;
    fb.open(file_path, std::ios::out | std::ios::binary);
    std::ostream out_stream(&fb);
    ply_file.write(out_stream, true); // binary format
}

// -----------------------------------------------------------------------------
//  COLMAP binary parsers
// -----------------------------------------------------------------------------
struct ImagePoint {
    double x, y;
    uint64_t point_id;
}; // skipped, definition for sizeof

std::vector<Image> read_images_binary(const std::filesystem::path& file_path) {
    auto buf_owner = read_binary(file_path);
    const char* cur = buf_owner->data();
    const char* end = cur + buf_owner->size();

    const uint64_t n_images = read_u64(cur);
    std::vector<Image> images;
    images.reserve(n_images);

    for (uint64_t i = 0; i < n_images; ++i) {
        const uint32_t id = read_u32(cur);
        auto& img = images.emplace_back(id);

        img._qvec.w() = static_cast<float>(read_f64(cur));
        img._qvec.x() = static_cast<float>(read_f64(cur));
        img._qvec.y() = static_cast<float>(read_f64(cur));
        img._qvec.z() = static_cast<float>(read_f64(cur));
        img._qvec.normalize();

        img._tvec.x() = static_cast<float>(read_f64(cur));
        img._tvec.y() = static_cast<float>(read_f64(cur));
        img._tvec.z() = static_cast<float>(read_f64(cur));

        img._camera_id = read_u32(cur);

        // null‑terminated filename
        img._name.assign(cur);
        cur += img._name.size() + 1;

        // skip 2‑D points
        const uint64_t npts = read_u64(cur);
        cur += npts * (sizeof(double) * 2 + sizeof(uint64_t));
    }

    if (cur != end)
        throw std::runtime_error("images.bin: unexpected trailing bytes");

    return images;
}

std::unordered_map<uint32_t, CameraInfo>
read_cameras_binary(const std::filesystem::path& file_path) {
    auto buf_owner = read_binary(file_path);
    const char* cur = buf_owner->data();
    const char* end = cur + buf_owner->size();

    const uint64_t n_cams = read_u64(cur);
    std::unordered_map<uint32_t, CameraInfo> cams;
    cams.reserve(n_cams);

    for (uint64_t i = 0; i < n_cams; ++i) {
        CameraInfo cam; // default ctor
        cam._camera_ID = read_u32(cur);
        const int32_t model_id = read_i32(cur);
        cam._width = read_u64(cur);
        cam._height = read_u64(cur);

        const auto it = camera_model_ids.find(model_id);
        if (it == camera_model_ids.end() || it->second.second < 0)
            throw std::runtime_error("Unsupported camera model id " + std::to_string(model_id));

        cam._camera_model = it->second.first;
        const int32_t param_cnt = it->second.second;
        cam._params.resize(param_cnt);
        std::memcpy(cam._params.data(), cur, param_cnt * sizeof(double));
        cur += param_cnt * sizeof(double);

        cams.emplace(cam._camera_ID, std::move(cam));
    }

    if (cur != end)
        throw std::runtime_error("cameras.bin: unexpected trailing bytes");

    return cams;
}

PointCloud read_point3D_binary(const std::filesystem::path& file_path) {
    auto buf_owner = read_binary(file_path);
    const char* cur = buf_owner->data();
    const char* end = cur + buf_owner->size();

    const uint64_t N = read_u64(cur);
    struct Packed {
        float x, y, z;
        uint8_t r, g, b;
    };
    std::vector<Packed> tmp;
    tmp.reserve(N);

    for (uint64_t i = 0; i < N; ++i) {
        cur += 8; // skip point id
        double dx = read_f64(cur), dy = read_f64(cur), dz = read_f64(cur);
        uint8_t r = *cur++, g = *cur++, b = *cur++;
        cur += 8; // reprojection error (ignored)
        const uint64_t tlen = read_u64(cur);
        cur += tlen * sizeof(uint32_t) * 2;
        tmp.push_back({float(dx), float(dy), float(dz), r, g, b});
    }

    if (cur != end)
        throw std::runtime_error("points3D.bin: unexpected trailing bytes");

    PointCloud pc;
    pc._points.resize(N);
    pc._colors.resize(N);
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(N); ++i) {
        const auto& p = tmp[i];
        pc._points[i] = {p.x, p.y, p.z};
        pc._colors[i] = {p.r, p.g, p.b};
    }
    return pc;
}

// -----------------------------------------------------------------------------
//  High‑level helpers
// -----------------------------------------------------------------------------
std::vector<CameraInfo> read_colmap_cameras(const std::filesystem::path file_path,
                                            const std::unordered_map<uint32_t, CameraInfo>& cameras,
                                            const std::vector<Image>& images,
                                            int resolution) {
    std::vector<CameraInfo> cam_infos(images.size());

    std::vector<std::future<void>> futures;
    futures.reserve(images.size());

    for (size_t idx = 0; idx < images.size(); ++idx) {
        const Image* img = &images[idx];
        auto it = cameras.find(img->_camera_id);
        if (it == cameras.end())
            throw std::runtime_error("Camera ID " + std::to_string(img->_camera_id) + " not found");

        cam_infos[idx] = it->second; // copy base parameters

        futures.emplace_back(std::async(
            std::launch::async,
            [=](const std::filesystem::path& images_root, const Image* image, CameraInfo* cam) {
                auto [data, w, h, c] = read_image(images_root / image->_name, resolution);
                cam->_img_w = w;
                cam->_img_h = h;
                cam->_channels = c;
                cam->_img_data = data;

                cam->_R = qvec2rotmat(image->_qvec);
                cam->_T = image->_tvec;
                cam->_image_name = image->_name;
                cam->_image_path = images_root / image->_name;

                switch (cam->_camera_model) {
                case CAMERA_MODEL::SIMPLE_PINHOLE: {
                    const float fx = cam->_params[0];
                    cam->_fov_x = focal2fov(fx, cam->_width);
                    cam->_fov_y = focal2fov(fx, cam->_height);
                    break;
                }
                case CAMERA_MODEL::PINHOLE: {
                    const float fx = cam->_params[0];
                    const float fy = cam->_params[1];
                    cam->_fov_x = focal2fov(fx, cam->_width);
                    cam->_fov_y = focal2fov(fy, cam->_height);
                    break;
                }
                default:
                    throw std::runtime_error("Camera model not supported in read_colmap_cameras");
                }
            },
            file_path, img, &cam_infos[idx]));
    }

    for (auto& f : futures)
        f.get();
    return cam_infos;
}

static std::pair<Eigen::Vector3f, float> get_center_and_diag(const std::vector<Eigen::Vector3f>& centers) {
    Eigen::Vector3f avg = Eigen::Vector3f::Zero();
    for (const auto& c : centers)
        avg += c;
    avg /= static_cast<float>(centers.size());

    float max_d = 0.f;
    for (const auto& c : centers)
        max_d = std::max(max_d, (c - avg).norm());

    return {avg, max_d};
}

float getNerfppNorm(std::vector<CameraInfo>& cams) {
    std::vector<Eigen::Vector3f> centers;
    centers.reserve(cams.size());
    for (auto& cam : cams) {
        Eigen::Matrix4f W2C = getWorld2View2Eigen(cam._R, cam._T);
        Eigen::Matrix4f C2W = W2C.inverse();
        centers.emplace_back(C2W.block<3, 1>(0, 3));
    }
    return get_center_and_diag(centers).second * 1.1f; // 10 % margin
}

std::unique_ptr<SceneInfo> read_colmap_scene_info(std::filesystem::path file_path, int resolution) {
    auto cameras = read_cameras_binary(file_path / "sparse/0/cameras.bin");
    auto images = read_images_binary(file_path / "sparse/0/images.bin");

    auto scene = std::make_unique<SceneInfo>();
    scene->_point_cloud = read_point3D_binary(file_path / "sparse/0/points3D.bin");
    scene->_cameras = read_colmap_cameras(file_path / "images", cameras, images, resolution);

    const auto& cam0 = scene->_cameras.front();
    const size_t n = scene->_cameras.size();
    const float mpix = cam0._img_w * cam0._img_h / 1'000'000.f;
    const bool resized = (resolution == 2 || resolution == 4 || resolution == 8);

    std::cout << "Training with " << n << " images of "
              << cam0._img_w << " x " << cam0._img_h
              << (resized ? " (resized) " : " ")
              << "pixels (" << std::fixed << std::setprecision(3) << mpix << " Mpixel per image, "
              << std::fixed << std::setprecision(1) << mpix * n << " Mpixel total)" << std::endl;

    scene->_nerf_norm_radius = getNerfppNorm(scene->_cameras);
    return scene;
}
