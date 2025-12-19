from PIL import Image
import pickle
import torch
import numpy as np
from typing import Optional


def _crop_resize_img(crop_size: int, img: Image.Image, img_is_mask: bool = False) -> Image.Image:
    outsize = crop_size
    short_size = outsize
    w, h = img.size
    if w > h:
        oh = short_size
        ow = int(1.0 * w * oh / h)
    else:
        ow = short_size
        oh = int(1.0 * h * ow / w)
    if not img_is_mask:
        img = img.resize((ow, oh), Image.BILINEAR)
    else:
        img = img.resize((ow, oh), Image.NEAREST)
    # center crop
    w, h = img.size
    x1 = int(round((w - outsize) / 2.))
    y1 = int(round((h - outsize) / 2.))
    img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
    return img


def _resize_img(crop_size: int, img: Image.Image, img_is_mask: bool = False) -> Image.Image:
    outsize = crop_size
    if not img_is_mask:
        img = img.resize((outsize, outsize), Image.BILINEAR)
    else:
        img = img.resize((outsize, outsize), Image.NEAREST)
    return img


def load_image_in_PIL_to_Tensor(path: str, split: str = 'train', mode: str = 'RGB', transform: Optional[object] = None, cfg: Optional[object] = None):
    img = Image.open(path).convert(mode)
    if cfg is not None and getattr(cfg, 'crop_img_and_mask', False):
        if split == 'train':
            img = _crop_resize_img(cfg.crop_size, img, img_is_mask=False)
        else:
            img = _resize_img(cfg.crop_size, img, img_is_mask=False)
    if transform:
        return transform(img)
    return img


def load_audio_lm(audio_lm_path: str):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    # If it's a torch tensor, detach and return
    if hasattr(audio_log_mel, 'detach'):
        try:
            return audio_log_mel.detach()
        except Exception:
            return audio_log_mel
    # If numpy array, convert to torch tensor
    if isinstance(audio_log_mel, np.ndarray):
        return torch.from_numpy(audio_log_mel)
    # Try to coerce to tensor for other array-likes
    try:
        return torch.as_tensor(audio_log_mel)
    except Exception:
        return audio_log_mel
