#include "core/colmap_reader.hpp"
#include "core/point_cloud.hpp"
#include "core/torch_shapes.hpp"
#include <algorithm>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

namespace F = torch::nn::functional;

// -----------------------------------------------------------------------------
//  Quaternion to rotation matrix
// -----------------------------------------------------------------------------
inline torch::Tensor qvec2rotmat(const torch::Tensor& qraw) {
    assert_vec(qraw, 4, "qvec");

    auto q = F::normalize(qraw.to(torch::kFloat32),
                          F::NormalizeFuncOptions().dim(0));

    auto w = q[0], x = q[1], y = q[2], z = q[3];

    torch::Tensor R = torch::empty({3, 3}, torch::kFloat32);
    R[0][0] = 1 - 2 * (y * y + z * z);
    R[0][1] = 2 * (x * y - z * w);
    R[0][2] = 2 * (x * z + y * w);

    R[1][0] = 2 * (x * y + z * w);
    R[1][1] = 1 - 2 * (x * x + z * z);
    R[1][2] = 2 * (y * z - x * w);

    R[2][0] = 2 * (x * z - y * w);
    R[2][1] = 2 * (y * z + x * w);
    R[2][2] = 1 - 2 * (x * x + y * y);
    return R;
}

inline float focal2fov(float focal, int pixels) {
    return 2.0f * std::atan(pixels / (2.0f * focal));
}

class Image {
public:
    Image() = default;
    explicit Image(uint32_t id)
        : _image_ID(id) {}

    uint32_t _camera_id = 0;
    std::string _name;

    torch::Tensor _qvec = torch::tensor({1.f, 0.f, 0.f, 0.f}, torch::kFloat32);
    torch::Tensor _tvec = torch::zeros({3}, torch::kFloat32);

private:
    uint32_t _image_ID = 0;
};

// -----------------------------------------------------------------------------
//  Build 4x4 world-to-camera matrix
// -----------------------------------------------------------------------------
inline torch::Tensor getWorld2View(const torch::Tensor& R,
                                   const torch::Tensor& T) {
    assert_mat(R, 3, 3, "R");
    assert_vec(T, 3, "T");

    torch::Tensor M = torch::eye(4, torch::kFloat32);
    M.index_put_({torch::indexing::Slice(0, 3),
                  torch::indexing::Slice(0, 3)},
                 R);
    M.index_put_({torch::indexing::Slice(0, 3), 3},
                 (-torch::matmul(R, T)).reshape({3}));
    return M;
}

// -----------------------------------------------------------------------------
//  POD read helpers
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
//  COLMAP camera-model map
// -----------------------------------------------------------------------------
static const std::unordered_map<int, std::pair<CAMERA_MODEL, int32_t>> camera_model_ids = {
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
//  Binary-file loader
// -----------------------------------------------------------------------------
static std::unique_ptr<std::vector<char>>
read_binary(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error("Failed to open " + p.string());

    auto sz = static_cast<std::streamsize>(f.tellg());
    auto buf = std::make_unique<std::vector<char>>(static_cast<size_t>(sz));

    f.seekg(0, std::ios::beg);
    f.read(buf->data(), sz);
    if (!f)
        throw std::runtime_error("Short read on " + p.string());
    return buf;
}

// -----------------------------------------------------------------------------
//  images.bin
// -----------------------------------------------------------------------------
std::vector<Image> read_images_binary(const std::filesystem::path& file_path) {
    auto buf_owner = read_binary(file_path);
    const char* cur = buf_owner->data();
    const char* end = cur + buf_owner->size();

    uint64_t n_images = read_u64(cur);
    std::vector<Image> images;
    images.reserve(n_images);

    for (uint64_t i = 0; i < n_images; ++i) {
        uint32_t id = read_u32(cur);
        auto& img = images.emplace_back(id);

        torch::Tensor q = torch::empty({4}, torch::kFloat32);
        for (int k = 0; k < 4; ++k)
            q[k] = static_cast<float>(read_f64(cur));

        img._qvec = q;

        torch::Tensor t = torch::empty({3}, torch::kFloat32);
        for (int k = 0; k < 3; ++k)
            t[k] = static_cast<float>(read_f64(cur));
        img._tvec = t;

        img._camera_id = read_u32(cur);

        img._name.assign(cur);
        cur += img._name.size() + 1; // skip '\0'

        uint64_t npts = read_u64(cur); // skip 2-D points
        cur += npts * (sizeof(double) * 2 + sizeof(uint64_t));
    }
    if (cur != end)
        throw std::runtime_error("images.bin: trailing bytes");
    return images;
}

// -----------------------------------------------------------------------------
//  cameras.bin
// -----------------------------------------------------------------------------
std::unordered_map<uint32_t, CameraData>
read_cameras_binary(const std::filesystem::path& file_path) {
    auto buf_owner = read_binary(file_path);
    const char* cur = buf_owner->data();
    const char* end = cur + buf_owner->size();

    uint64_t n_cams = read_u64(cur);
    std::unordered_map<uint32_t, CameraData> cams;
    cams.reserve(n_cams);

    for (uint64_t i = 0; i < n_cams; ++i) {
        CameraData cam;
        cam._camera_ID = read_u32(cur);

        int32_t model_id = read_i32(cur);
        cam._width = read_u64(cur);
        cam._height = read_u64(cur);

        auto it = camera_model_ids.find(model_id);
        if (it == camera_model_ids.end() || it->second.second < 0)
            throw std::runtime_error("Unsupported camera-model id " + std::to_string(model_id));

        cam._camera_model = it->second.first;
        int32_t param_cnt = it->second.second;
        cam._params = torch::from_blob(const_cast<char*>(cur),
                                       {param_cnt}, torch::kFloat64)
                          .clone()
                          .to(torch::kFloat32);
        cur += param_cnt * sizeof(double);

        cams.emplace(cam._camera_ID, std::move(cam));
    }
    if (cur != end)
        throw std::runtime_error("cameras.bin: trailing bytes");
    return cams;
}

// -----------------------------------------------------------------------------
//  points3D.bin
// -----------------------------------------------------------------------------
PointCloud read_point3D_binary(const std::filesystem::path& file_path) {
    auto buf_owner = read_binary(file_path);
    const char* cur = buf_owner->data();
    const char* end = cur + buf_owner->size();

    uint64_t N = read_u64(cur);

    // Pre-allocate tensors directly
    torch::Tensor positions = torch::empty({static_cast<int64_t>(N), 3}, torch::kFloat32);
    torch::Tensor colors = torch::empty({static_cast<int64_t>(N), 3}, torch::kUInt8);

    // Get raw pointers for efficient access
    float* pos_data = positions.data_ptr<float>();
    uint8_t* col_data = colors.data_ptr<uint8_t>();

    for (uint64_t i = 0; i < N; ++i) {
        cur += 8; // skip point ID

        // Read position directly into tensor
        pos_data[i * 3 + 0] = static_cast<float>(read_f64(cur));
        pos_data[i * 3 + 1] = static_cast<float>(read_f64(cur));
        pos_data[i * 3 + 2] = static_cast<float>(read_f64(cur));

        // Read color directly into tensor
        col_data[i * 3 + 0] = *cur++;
        col_data[i * 3 + 1] = *cur++;
        col_data[i * 3 + 2] = *cur++;

        cur += 8;                                    // skip reprojection error
        cur += read_u64(cur) * sizeof(uint32_t) * 2; // skip track
    }

    if (cur != end)
        throw std::runtime_error("points3D.bin: trailing bytes");

    return PointCloud(positions, colors);
}

// -----------------------------------------------------------------------------
//  Assemble per-image camera information
// -----------------------------------------------------------------------------
std::tuple<std::vector<CameraData>, torch::Tensor>
read_colmap_cameras(const std::filesystem::path base_path,
                    const std::unordered_map<uint32_t, CameraData>& cams,
                    const std::vector<Image>& images,
                    const std::string& images_folder = "images") {
    std::vector<CameraData> out(images.size());

    std::filesystem::path images_path = base_path / images_folder;

    // Prepare tensor to store all camera locations [N, 3]
    torch::Tensor camera_locations = torch::zeros({static_cast<int64_t>(images.size()), 3}, torch::kFloat32);

    // Check if the specified images folder exists
    if (!std::filesystem::exists(images_path)) {
        throw std::runtime_error("Images folder does not exist: " + images_path.string());
    }

    for (size_t i = 0; i < images.size(); ++i) {
        const Image& img = images[i];
        auto it = cams.find(img._camera_id);
        if (it == cams.end())
            throw std::runtime_error("Camera ID " + std::to_string(img._camera_id) + " not found");

        out[i] = it->second;
        out[i]._image_path = images_path / img._name;
        out[i]._image_name = img._name;

        out[i]._R = qvec2rotmat(img._qvec);
        out[i]._T = img._tvec.clone();

        // Camera location in world space = -R^T * T
        // This is equivalent to extracting camtoworlds[:, :3, 3] after inverting w2c
        camera_locations[i] = -torch::matmul(out[i]._R.t(), out[i]._T);

        switch (out[i]._camera_model) {
        case CAMERA_MODEL::SIMPLE_PINHOLE: {
            float fx = out[i]._params[0].item<float>();
            out[i]._fov_x = focal2fov(fx, out[i]._width);
            out[i]._fov_y = focal2fov(fx, out[i]._height);
            break;
        }
        case CAMERA_MODEL::PINHOLE: {
            float fx = out[i]._params[0].item<float>();
            float fy = out[i]._params[1].item<float>();
            out[i]._fov_x = focal2fov(fx, out[i]._width);
            out[i]._fov_y = focal2fov(fy, out[i]._height);
            break;
        }
        default:
            throw std::runtime_error("Unsupported camera model");
        }

        out[i]._img_w = out[i]._img_h = out[i]._channels = 0;
        out[i]._img_data = nullptr;
    }

    std::cout << "Training with " << out.size() << " images \n";
    return {std::move(out), camera_locations.mean(0)};
}

// -----------------------------------------------------------------------------
//  Public API functions
// -----------------------------------------------------------------------------
PointCloud read_colmap_point_cloud(const std::filesystem::path& filepath) {
    return read_point3D_binary(filepath / "sparse/0/points3D.bin");
}

std::tuple<std::vector<CameraData>, torch::Tensor> read_colmap_cameras_and_images(
    const std::filesystem::path& base,
    const std::string& images_folder) {

    auto cams = read_cameras_binary(base / "sparse/0/cameras.bin");
    auto images = read_images_binary(base / "sparse/0/images.bin");

    return read_colmap_cameras(base, cams, images, images_folder);
}
