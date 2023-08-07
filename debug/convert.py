import torch
import numpy as np
from torchvision import transforms
from PIL import Image

tensor_specs = {
    "image": {"dims": 3, "shape": [3, 546, 979], "type": float},
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


for i in range(251):
    tensor = load_tensor(f"{i}.tensor", tensor_specs["image"])
    tensor_spec = {
        "dims": len(tensor.shape),
        "shape": tuple(tensor.shape),
        "type": float
    }
    img = transforms.ToPILImage()(tensor)
    img.save(f"{i}_img.png")