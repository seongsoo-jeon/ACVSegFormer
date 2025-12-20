import pickle
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import importlib.util
import importlib.machinery
import sys
from pathlib import Path
import types

# Load io_utils first and register package entries so relative imports work
base = Path(__file__).parents[1]
io_utils_path = str(base / 'dataloader' / 'io_utils.py')
loader = importlib.machinery.SourceFileLoader('dataloader.io_utils', io_utils_path)
spec = importlib.util.spec_from_loader(loader.name, loader)
io_utils = importlib.util.module_from_spec(spec)
sys.modules['dataloader'] = types.ModuleType('dataloader')
sys.modules['dataloader.io_utils'] = io_utils
loader.exec_module(io_utils)

def _load_submodule(name: str, filename: str):
    path = str(base / 'dataloader' / filename)
    loader = importlib.machinery.SourceFileLoader(f'dataloader.{name}', path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = mod
    loader.exec_module(mod)
    return mod

ms3_dataset = _load_submodule('ms3_dataset', 'ms3_dataset.py')
s4_dataset = _load_submodule('s4_dataset', 's4_dataset.py')
v2_dataset = _load_submodule('v2_dataset', 'v2_dataset.py')


def test_load_image_pil_and_tensor(tmp_path):
    img_path = tmp_path / "img.png"
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    img.save(img_path)

    out = io_utils.load_image_in_PIL_to_Tensor(str(img_path))
    assert isinstance(out, Image.Image)

    t = transforms.ToTensor()
    out2 = io_utils.load_image_in_PIL_to_Tensor(str(img_path), transform=t)
    assert isinstance(out2, torch.Tensor)
    assert out2.shape[0] == 3


def test_load_audio_lm_types(tmp_path):
    # torch tensor
    t = torch.randn(2, 3)
    p1 = tmp_path / "t.pkl"
    with open(p1, 'wb') as f:
        pickle.dump(t, f)
    out = io_utils.load_audio_lm(str(p1))
    assert isinstance(out, torch.Tensor)

    # numpy array -> expect torch.Tensor (io_utils converts numpy to tensor)
    a = np.zeros((5, 4))
    p2 = tmp_path / "a.pkl"
    with open(p2, 'wb') as f:
        pickle.dump(a, f)
    out2 = io_utils.load_audio_lm(str(p2))
    assert isinstance(out2, torch.Tensor)

    # generic object -> returned as-is
    obj = {"x": 1}
    p3 = tmp_path / "o.pkl"
    with open(p3, 'wb') as f:
        pickle.dump(obj, f)
    out3 = io_utils.load_audio_lm(str(p3))
    assert out3 == obj


def test_dataset_classes_exist():
    from torch.utils.data import Dataset

    assert issubclass(ms3_dataset.MS3Dataset, Dataset)
    assert issubclass(s4_dataset.S4Dataset, Dataset)
    assert issubclass(v2_dataset.V2Dataset, Dataset)
