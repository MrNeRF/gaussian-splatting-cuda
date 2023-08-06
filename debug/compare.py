import torch
import numpy as np

# Specification for tensors
tensor_specs = {
    "image": {"dims": 3, "shape": [3, 546, 979], "type": float},
    "means3D": {"dims": 2, "shape": [136029, 3], "type": float},
    "sh": {"dims": 3, "shape": [136029, 16, 3], "type": float},
    "colors_precomp": {"dims": 1, "shape": [0], "type": float},
    "opacities": {"dims": 2, "shape": [136029, 1], "type": float},
    "scales": {"dims": 2, "shape": [136029, 3], "type": float},
    "rotations": {"dims": 2, "shape": [136029, 4], "type": float},
    "cov3Ds_precomp": {"dims": 1, "shape": [0], "type": float},
    "image_height": {"dims": 0, "shape": [], "type": np.int64},
    "image_width": {"dims": 0, "shape": [], "type": np.int64},
    "tanfovx": {"dims": 0, "shape": [], "type": float},
    "tanfovy": {"dims": 0, "shape": [], "type": float},
    "bg": {"dims": 1, "shape": [3], "type": float},
    "scale_modifier": {"dims": 0, "shape": [], "type": float},
    "viewmatrix": {"dims": 2, "shape": [4, 4], "type": float},
    "projmatrix": {"dims": 2, "shape": [4, 4], "type": float},
    "sh_degree": {"dims": 0, "shape": [], "type": np.int64},
    "camera_center": {"dims": 1, "shape": [3], "type": float},
    "prefiltered": {"dims": 0, "shape": [], "type": bool},
    "max_radii2D": {"dims": 1, "shape": [136029], "type": float},
    "visibility_filter": {"dims": 1, "shape": [136029], "type": bool},
    "radii": {"dims": 1, "shape": [136029], "type": int},
    "viewspace_point_tensor": {"dims": 2, "shape": [136029, 3], "type": float},
    "max_radii2D_masked": {"dims": 1, "shape": [136029], "type": float},
}


def load_tensor(filename, tensor_spec):
    with open(filename, 'rb') as f:
        dims = int.from_bytes(f.read(4), 'little')
        shape = tuple(int.from_bytes(f.read(8), 'little') for _ in range(dims))
        assert dims == tensor_spec["dims"], f"Expected dims {tensor_spec['dims']} for {filename}, got {dims}"
        assert shape == tuple(tensor_spec["shape"]), f"Expected shape {tensor_spec['shape']} for {filename}, got {shape}"
        
        data_type = tensor_spec["type"]
        
        if data_type == bool:
            data = np.fromfile(f, dtype=np.bool_).astype(np.bool_)
        elif data_type == float:
            data = np.fromfile(f, dtype=np.float32).astype(np.float32)
        elif data_type == int:
            data = np.fromfile(f, dtype=np.int32).astype(np.int32)
        else:
            data = np.fromfile(f, dtype=np.int64).astype(np.int64)
        
        # Reshape the data based on tensor specification, unless it's a scalar
        if tensor_spec["dims"] != 0:
            print(f"Filename: {filename}")
            print(f"Total size of loaded data: {data.size}")
            print(f"Expected shape from tensor_spec: {tensor_spec['shape']}")
            data = data.reshape(tensor_spec["shape"])
        
        return torch.from_numpy(data)


py_tensors = {
    name: torch.load(f"pytorch_{name}.pt", map_location="cpu") for name in tensor_specs.keys()
}

libtorch_tensors = {
    name: load_tensor(f"libtorch_{name}.pt", tensor_specs[name]) for name in tensor_specs.keys()
}

tolerance = 1e-5

for name, tensor in py_tensors.items():
    print(f"======= Comparing {name} =======")
    libtorch_tensor = libtorch_tensors.get(name, None)
    
    if libtorch_tensor is None:
        print(f"{name}: libtorch tensor is None!")
        continue

    if tensor is None:
        print(f"{name}: pytorch tensor is None!")
        print(f"{name} (libtorch): Shape: {libtorch_tensor.shape}")
        if torch.any(libtorch_tensor != 0):
            print(f"{name} (libtorch): Contains non-zero values!")
        else:
            print(f"{name} (libtorch): All values are zero!")
        continue

    # Check for shape mismatches
    if libtorch_tensor.shape != tensor.shape:
        print(f"Shape mismatch for {name}: "
              f"Expected {tensor.shape} but got {libtorch_tensor.shape}")
        continue
    
    approx_equal = torch.isclose(libtorch_tensor, tensor, atol=tolerance)
    
    if not torch.all(approx_equal):
        print(f"Value mismatch for {name}")
        
        # Get indices of mismatched values
        mismatched_indices = torch.nonzero(~approx_equal, as_tuple=True)
        
        # Get the mismatched values from both tensors
        mismatched_pytorch_values = tensor[mismatched_indices]
        mismatched_libtorch_values = libtorch_tensor[mismatched_indices]
        
        # Show up to 20 of the mismatched values side by side
        print(f"Showing up to 20 of {mismatched_pytorch_values.shape[0]} mismatched values")
        num_to_show = min(20, mismatched_pytorch_values.shape[0])
        for i in range(num_to_show):
            print(f"Index: {mismatched_indices[0][i]}, PyTorch: {mismatched_pytorch_values[i]}, LibTorch: {mismatched_libtorch_values[i]}")
    else:
        print(f"{name} matches!")
