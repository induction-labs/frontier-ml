"""Read ordinary checkpoint metadata without executing submission code.

PyTorch uses its restricted weights-only loader and fake tensors. NumPy never
unpickles object arrays; ONNX external-data paths are recorded but never opened.
Unknown encodings are explicit evidence gaps, not automatic task violations.
"""
import ast
import math
from pathlib import Path
import struct
import zipfile


def _torch_inventory(path):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    tensors = {}

    def visit(value, name):
        if isinstance(value, torch.Tensor):
            tensors[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                             "elements": value.numel(), "stored_bytes": value.numel() * value.element_size()}
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{name}.{key}" if name else str(key))
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, f"{name}[{index}]")

    visit(checkpoint, "")
    return {"format": "pytorch", "inspection": "complete", "tensors": tensors}


def _numpy_inventory(path):
    import numpy as np

    def header(stream, size):
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=size)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=size)
        elif version == (3, 0):
            # Version 3 uses the v2 length field with a UTF-8 descriptor.
            length, = struct.unpack("<I", stream.read(4))
            if length > size - stream.tell():
                raise ValueError("Truncated NumPy header")
            fields = ast.literal_eval(stream.read(length).decode("utf-8"))
            if not isinstance(fields, dict) or set(fields) != {"shape", "fortran_order", "descr"}:
                raise ValueError("Invalid NumPy header")
            shape, fortran, dtype = fields["shape"], fields["fortran_order"], np.dtype(fields["descr"])
        else:
            raise ValueError(f"Unknown NumPy version {version}")
        if not isinstance(shape, tuple) or any(type(n) is not int or n < 0 for n in shape) or type(fortran) is not bool:
            raise ValueError("Invalid NumPy dimensions or order")
        if dtype.hasobject:
            raise ValueError("Object arrays require pickle and are not read by the trusted inspector")
        count = math.prod(shape)
        if stream.tell() + count * dtype.itemsize > size:
            raise ValueError("Truncated NumPy array")
        return {"shape": list(shape), "dtype": str(dtype), "elements": count,
                "stored_bytes": count * dtype.itemsize, "fortran_order": fortran}

    if path.suffix.lower() == ".npz":
        with zipfile.ZipFile(path) as archive:
            tensors = {}
            for entry in archive.infolist():
                if not entry.is_dir() and entry.filename.endswith(".npy"):
                    with archive.open(entry) as stream:
                        tensors[entry.filename.removesuffix(".npy")] = header(stream, entry.file_size)
    else:
        with path.open("rb") as stream:
            tensors = {path.stem: header(stream, path.stat().st_size)}
    return {"format": "numpy", "inspection": "complete", "tensors": tensors}


def _onnx_inventory(path):
    import onnx

    model = onnx.load_model(str(path), load_external_data=False)
    tensors = {}

    def tensor(value, name):
        tensors[name] = {"shape": list(value.dims), "dtype": onnx.TensorProto.DataType.Name(value.data_type),
                         "elements": math.prod(value.dims)}
        if value.external_data:
            tensors[name]["external_data"] = {entry.key: entry.value for entry in value.external_data}

    operators = {}

    def graph(value, prefix=""):
        for index, value_tensor in enumerate(value.initializer):
            tensor(value_tensor, prefix + (value_tensor.name or f"initializer_{index}"))
        for index, node in enumerate(value.node):
            op = f"{node.domain}:{node.op_type}"
            operators[op] = operators.get(op, 0) + 1
            for attr in node.attribute:
                name = f"{prefix}node_{index}/{attr.name}"
                if attr.type == onnx.AttributeProto.TENSOR:
                    tensor(attr.t, name)
                elif attr.type == onnx.AttributeProto.TENSORS:
                    for i, item in enumerate(attr.tensors):
                        tensor(item, f"{name}/{i}")
                elif attr.type == onnx.AttributeProto.GRAPH:
                    graph(attr.g, name + "/")
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for i, item in enumerate(attr.graphs):
                        graph(item, f"{name}/{i}/")

    graph(model.graph)
    return {"format": "onnx", "inspection": "complete", "tensors": tensors, "operators": operators}


def checkpoint_inventory(path):
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix in {".pt", ".pth", ".bin"}:
            result = _torch_inventory(path)
        elif suffix in {".npy", ".npz"}:
            result = _numpy_inventory(path)
        elif suffix == ".onnx":
            result = _onnx_inventory(path)
        else:
            raise ValueError("No safe metadata reader for this encoding")
    except Exception as error:
        result = {"format": suffix.lstrip("."), "inspection": "unavailable", "tensors": {},
                  "reason": f"{type(error).__name__}: {error}"}
    return {"bytes": path.stat().st_size, **result}
