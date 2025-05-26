// Copyright (c) 2023 Janusch Patas.
#pragma once

#include "core/debug_utils.hpp"
#include <fstream>
#include <iostream>
#include <torch/torch.h>
#include <tuple>

// Serialize a single tensor to an output stream
inline void serializeTensor(std::ostream& os, const torch::Tensor& tensor) {
    //    ts::print_debug_info(tensor, "tensor");
    if (!tensor.defined()) {
        // Handle error: e.g., throw an exception or return early
        throw std::runtime_error("Tensor is not defined!");
    }

    auto cpu_tensor = tensor.to(torch::kCPU); // Move tensor to CPU
    // Write the tensor's scalar type to the stream first
    int64_t scalar_type = static_cast<int64_t>(cpu_tensor.scalar_type());
    os.write(reinterpret_cast<const char*>(&scalar_type), sizeof(int64_t));

    int64_t dims = cpu_tensor.dim();
    os.write(reinterpret_cast<const char*>(&dims), sizeof(int64_t));

    for (int64_t i = 0; i < dims; i++) {
        int64_t size = cpu_tensor.size(i);
        os.write(reinterpret_cast<const char*>(&size), sizeof(int64_t));
    }

    int64_t num_elements = cpu_tensor.numel();
    auto flat_tensor = cpu_tensor.contiguous().view({num_elements});

    // Serialize based on data type
    switch (cpu_tensor.scalar_type()) {
    case torch::kFloat:
        os.write(reinterpret_cast<const char*>(flat_tensor.data_ptr<float>()), num_elements * sizeof(float));
        break;
    case torch::kInt64:
        os.write(reinterpret_cast<const char*>(flat_tensor.data_ptr<int64_t>()), num_elements * sizeof(int64_t));
        break;
    case torch::kInt32:
        os.write(reinterpret_cast<const char*>(flat_tensor.data_ptr<int32_t>()), num_elements * sizeof(int32_t));
        break;
    case torch::kBool:
        os.write(reinterpret_cast<const char*>(flat_tensor.data_ptr<bool>()), num_elements * sizeof(bool));
        break;
    case torch::kByte: // For unsigned char
        os.write(reinterpret_cast<const char*>(flat_tensor.data_ptr<unsigned char>()), num_elements * sizeof(unsigned char));
        break;
    default:
        throw std::runtime_error("Unsupported tensor data type for serialization!");
    }
}

// Deserialize a tensor from an input stream
inline torch::Tensor deserializeTensor(std::istream& is) {
    // Read the tensor's scalar type first
    int64_t scalar_type;
    is.read(reinterpret_cast<char*>(&scalar_type), sizeof(int64_t));

    int64_t dims;
    is.read(reinterpret_cast<char*>(&dims), sizeof(int64_t));

    std::vector<int64_t> sizes(dims);
    for (int64_t i = 0; i < dims; i++) {
        is.read(reinterpret_cast<char*>(&sizes[i]), sizeof(int64_t));
    }

    torch::ScalarType dtype = static_cast<torch::ScalarType>(scalar_type);
    torch::Tensor tensor = torch::empty(sizes, dtype);

    // Deserialize based on data type
    switch (dtype) {
    case torch::kFloat:
        is.read(reinterpret_cast<char*>(tensor.data_ptr<float>()), tensor.numel() * sizeof(float));
        break;
    case torch::kInt64:
        is.read(reinterpret_cast<char*>(tensor.data_ptr<int64_t>()), tensor.numel() * sizeof(int64_t));
        break;
    case torch::kInt32:
        is.read(reinterpret_cast<char*>(tensor.data_ptr<int32_t>()), tensor.numel() * sizeof(int32_t));
        break;
    case torch::kBool:
        is.read(reinterpret_cast<char*>(tensor.data_ptr<bool>()), tensor.numel() * sizeof(bool));
        break;
    case torch::kByte: // For unsigned char
        is.read(reinterpret_cast<char*>(tensor.data_ptr<unsigned char>()), tensor.numel() * sizeof(unsigned char));
        break;
    default:
        throw std::runtime_error("Unsupported tensor data type for deserialization!");
    }
    return tensor.to(torch::kCUDA);
}

inline void serializeFloat(std::ostream& os, const float value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(float));
}

inline float deserializeFloat(std::istream& is) {
    float value;
    is.read(reinterpret_cast<char*>(&value), sizeof(float));
    return value;
}

inline void serializeInt(std::ostream& os, const int value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(int));
}

inline int deserializeInt(std::istream& is) {
    int value;
    is.read(reinterpret_cast<char*>(&value), sizeof(int));
    return value;
}

inline bool deserializeBool(std::istream& is) {
    int bool_as_int;
    is.read(reinterpret_cast<char*>(&bool_as_int), sizeof(int));
    return bool_as_int == 1;
}

inline void saveFunctionData(const std::string& filename,
                             const torch::Tensor& grad_means2D,
                             const torch::Tensor& grad_colors_precomp,
                             const torch::Tensor& grad_opacities,
                             const torch::Tensor& grad_means3D,
                             const torch::Tensor& grad_cov3Ds_precomp,
                             const torch::Tensor& grad_sh,
                             const torch::Tensor& grad_scales,
                             const torch::Tensor& grad_rotations,
                             const torch::Tensor& background,
                             const torch::Tensor& means3D,
                             const torch::Tensor& radii,
                             const torch::Tensor& colors,
                             const torch::Tensor& scales,
                             const torch::Tensor& rotations,
                             const float scale_modifier,
                             const torch::Tensor& cov3D_precomp,
                             const torch::Tensor& viewmatrix,
                             const torch::Tensor& projmatrix,
                             const float tan_fovx,
                             const float tan_fovy,
                             const torch::Tensor& dL_dout_color,
                             const torch::Tensor& sh,
                             const int degree,
                             const torch::Tensor& campos,
                             const torch::Tensor& geomBuffer,
                             const int R,
                             const torch::Tensor& binningBuffer,
                             const torch::Tensor& imageBuffer) {
    std::ofstream ofs(filename, std::ios::binary | std::ios::trunc);

    serializeTensor(ofs, grad_means2D);
    serializeTensor(ofs, grad_colors_precomp);
    serializeTensor(ofs, grad_opacities);
    serializeTensor(ofs, grad_means3D);
    serializeTensor(ofs, grad_cov3Ds_precomp);
    serializeTensor(ofs, grad_sh);
    serializeTensor(ofs, grad_scales);
    serializeTensor(ofs, grad_rotations);
    serializeTensor(ofs, background);
    serializeTensor(ofs, means3D);
    serializeTensor(ofs, radii);
    serializeTensor(ofs, colors);
    serializeTensor(ofs, scales);
    serializeTensor(ofs, rotations);
    serializeFloat(ofs, scale_modifier);
    serializeTensor(ofs, cov3D_precomp);
    serializeTensor(ofs, viewmatrix);
    serializeTensor(ofs, projmatrix);
    serializeFloat(ofs, tan_fovx);
    serializeFloat(ofs, tan_fovy);
    serializeTensor(ofs, dL_dout_color);
    serializeTensor(ofs, sh);
    serializeInt(ofs, degree);
    serializeTensor(ofs, campos);
    serializeTensor(ofs, geomBuffer);
    serializeInt(ofs, R);
    serializeTensor(ofs, binningBuffer);
    serializeTensor(ofs, imageBuffer);
    ofs.close();
}

inline void loadFunctionData(const std::string& filename,
                             torch::Tensor& grad_means2D,
                             torch::Tensor& grad_colors_precomp,
                             torch::Tensor& grad_opacities,
                             torch::Tensor& grad_means3D,
                             torch::Tensor& grad_cov3Ds_precomp,
                             torch::Tensor& grad_sh,
                             torch::Tensor& grad_scales,
                             torch::Tensor& grad_rotations,
                             torch::Tensor& background,
                             torch::Tensor& means3D,
                             torch::Tensor& radii,
                             torch::Tensor& colors,
                             torch::Tensor& scales,
                             torch::Tensor& rotations,
                             float& scale_modifier,
                             torch::Tensor& cov3D_precomp,
                             torch::Tensor& viewmatrix,
                             torch::Tensor& projmatrix,
                             float& tan_fovx,
                             float& tan_fovy,
                             torch::Tensor& dL_dout_color,
                             torch::Tensor& sh,
                             int& degree,
                             torch::Tensor& campos,
                             torch::Tensor& geomBuffer,
                             int& R,
                             torch::Tensor& binningBuffer,
                             torch::Tensor& imageBuffer) {
    std::ifstream ifs(filename, std::ios::binary);

    // Deserialize outputs
    grad_means2D = deserializeTensor(ifs);
    grad_colors_precomp = deserializeTensor(ifs);
    grad_opacities = deserializeTensor(ifs);
    grad_means3D = deserializeTensor(ifs);
    grad_cov3Ds_precomp = deserializeTensor(ifs);
    grad_sh = deserializeTensor(ifs);
    grad_scales = deserializeTensor(ifs);
    grad_rotations = deserializeTensor(ifs);
    background = deserializeTensor(ifs);
    means3D = deserializeTensor(ifs);
    radii = deserializeTensor(ifs);
    colors = deserializeTensor(ifs);
    scales = deserializeTensor(ifs);
    rotations = deserializeTensor(ifs);
    scale_modifier = deserializeFloat(ifs);
    cov3D_precomp = deserializeTensor(ifs);
    viewmatrix = deserializeTensor(ifs);
    projmatrix = deserializeTensor(ifs);
    tan_fovx = deserializeFloat(ifs);
    tan_fovy = deserializeFloat(ifs);
    dL_dout_color = deserializeTensor(ifs);
    sh = deserializeTensor(ifs);
    degree = deserializeInt(ifs);
    campos = deserializeTensor(ifs);
    geomBuffer = deserializeTensor(ifs);
    R = deserializeInt(ifs);
    binningBuffer = deserializeTensor(ifs);
    imageBuffer = deserializeTensor(ifs);
    ifs.close();
}
