# 2024 ARMLab - DenseTact Calibration (Streaming + Multi-GPU Ready)

import os
import json
import random
from itertools import islice

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


# -----------------------------
# Constants / normalization
# -----------------------------
DISP_MEANS  = np.array([-0.17508025467395782, -0.8888664841651917, -0.12558214366436005], dtype=np.float32)
FORCE_MEANS = np.array([-0.0026451176963746548, -0.010370887815952301, -0.002843874040991068], dtype=np.float32)


def get_densetact_dataset(
    mode,
    samples_roots,
    output_types,
    transform_X=None,
    transform_y=None,
    is_mae=False,
    normalization=False,
    contiguous_on_direction=False,
    shuffle_buffer=8192,
    epoch_size=None,
    rng_seed=42,
    split_mod=None,
    split_remainders=None,
):
    """
    Create a DenseTact dataset in either streaming or map-style mode.
    """
    if transform_X is None:
        transform_X = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True),
        ])
    if transform_y is None:
        transform_y = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True),
        ])

    if is_mae:
        output_types = []

    if mode == "stream":
        ds = DenseTactStream(
            samples_roots=samples_roots,
            output_types=output_types,
            transform_X=transform_X,
            transform_y=transform_y,
            normalization=normalization,
            contiguous_on_direction=contiguous_on_direction,
            shuffle_buffer=shuffle_buffer,
            epoch_size=epoch_size,
            rng_seed=rng_seed,
            is_mae=is_mae,
        )


        return ds
    else:
        raise ValueError(f"Unsupported dataset mode: {mode}")

# -----------------------------
# Helpers: listing, sharding, shuffle
# -----------------------------
def list_sample_dirs(roots):
    """Yield (n, xdir, ydir) for all X{n}/y{n} pairs under one or many roots."""
    if isinstance(roots, str):
        roots = [roots]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if not name.startswith("X"):
                continue
            n = name[1:]
            xdir = os.path.join(root, f"X{n}")
            ydir = os.path.join(root, f"y{n}")
            if os.path.isdir(xdir) and os.path.isdir(ydir):
                yield int(n), xdir, ydir


def shard_for_worker_and_rank(iterable):
    """
    Shard an iterable by DataLoader worker and DDP rank.
    """
    wi = get_worker_info()
    num_workers = wi.num_workers if wi else 1
    worker_id = wi.id if wi else 0

    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0

    # Composite index space: each worker in each rank is unique
    worker_global_id = rank * num_workers + worker_id
    total_workers = world_size * num_workers

    for i, x in enumerate(iterable):
        if i % total_workers == worker_global_id:
            yield x


class BufferedShuffle:
    """Streaming shuffle with bounded memory."""
    def __init__(self, iterable, bufsize=4096, rng=None):
        self.iterable = iterable
        self.bufsize = bufsize
        self.rng = rng or random.Random()

    def __iter__(self):
        buf = []
        for x in self.iterable:
            buf.append(x)
            if len(buf) >= self.bufsize:
                self.rng.shuffle(buf)
                while buf:
                    yield buf.pop()
        self.rng.shuffle(buf)
        while buf:
            yield buf.pop()


# -----------------------------
# Optional: build output mask once
# -----------------------------
def build_output_mask(samples_root, example_idx=111, out_size=None):
    """
    Mask = 1 where stress1 != cnorm, else 0. Returns float32 numpy array in [0,1].
    """
    ydir = os.path.join(samples_root, f"y{example_idx}")
    cnorm_p   = os.path.join(ydir, "cnorm.png")
    stress1_p = os.path.join(ydir, "stress1.png")
    if not (os.path.exists(cnorm_p) and os.path.exists(stress1_p)):
        return None

    cnorm = cv2.cvtColor(cv2.imread(cnorm_p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    stress1 = cv2.cvtColor(cv2.imread(stress1_p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    diff = (stress1.astype(np.int32) - cnorm.astype(np.int32))
    mask = (np.any(diff != 0, axis=-1)).astype(np.float32)  # 1 where different

    if out_size is not None:
        mask = cv2.resize(mask, out_size[::-1], interpolation=cv2.INTER_LINEAR)

    return mask


# -----------------------------
# Streaming Dataset
# -----------------------------
class DenseTactStream(IterableDataset):
    """
    Streams (X, y) from X{n}/y{n} folders.
    """
    VALID_OUTPUTS = ['depth', 'cnorm', 'stress1', 'stress2', 'disp', 'shear']

    def __init__(
        self,
        samples_roots,
        output_types,
        transform_X,
        transform_y,
        output_mask=None,
        shuffle_buffer=8192,
        epoch_size=None,
        rng_seed=42,
        normalization=False,
        contiguous_on_direction=False,
        is_mae=False
    ):
        super().__init__()
        for t in output_types:
            assert t in self.VALID_OUTPUTS, f"Invalid output type {t}"
        self.samples_roots = samples_roots
        self.output_types = output_types
        self.transform_X = transform_X
        self.transform_y = transform_y
        self.output_mask = output_mask
        self.shuffle_buffer = shuffle_buffer
        self.epoch_size = epoch_size
        self.rng_seed = rng_seed
        self.normalization = normalization
        self.contiguous_on_direction = contiguous_on_direction
        self.is_mae = is_mae

        # Count total samples (global) if possible
        self._total_samples = len(list(list_sample_dirs(samples_roots)))

    # ---------- low-level IO ----------
    def _read_rgb(self, p):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(p)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _first_existing(self, ydir, names):
        for nm in names:
            p = os.path.join(ydir, nm)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"None of {names} exist in {ydir}")

    def _bounds_for(self, bounds, prefer_key, fallback_key=None):
        if bounds is None:
            return None
        if prefer_key in bounds:
            return bounds[prefer_key]
        if fallback_key and fallback_key in bounds:
            return bounds[fallback_key]
        if prefer_key.upper() in bounds:
            return bounds[prefer_key.upper()]
        return None

    def _load_X(self, xdir):
        d = self._read_rgb(os.path.join(xdir, "deformed.png"))
        u = self._read_rgb(os.path.join(xdir, "undeformed.png"))
        x = np.concatenate([d, u], axis=2)          # (H,W,6), uint8
        return self.transform_X(x).float()

    @staticmethod
    def _decode_3ch(img_uint8, bk):
        img = img_uint8.astype(np.float32)
        r = (img[..., 0] / 255.0) * (bk['max_val_r'] - bk['min_val_r']) + bk['min_val_r']
        g = (img[..., 1] / 255.0) * (bk['max_val_g'] - bk['min_val_g']) + bk['min_val_g']
        b = (img[..., 2] / 255.0) * (bk['max_val_b'] - bk['min_val_b']) + bk['min_val_b']
        return np.stack([r, g, b], axis=2)

    def _load_y_numpy(self, ydir):
        bpath = os.path.join(ydir, "bounds.json")
        bounds = json.load(open(bpath)) if os.path.exists(bpath) else None

        parts = []
        for t in self.output_types:
            if t == 'depth':
                dp = cv2.imread(os.path.join(ydir, "depth.png"), cv2.IMREAD_ANYDEPTH)
                dp = (dp.astype(np.float32) / 10000.0) - 1.0
                parts.append(dp[..., None])

            elif t in ['cnorm', 'stress1', 'stress2', 'disp', 'shear']:
                name_map = {
                    'cnorm': ["cnforce_local.png", "cnorm.png"],
                    'stress1': ["nforce_local.png", "stress1.png"],
                    'stress2': ["sforce_local.png", "stress2.png"],
                    'disp': ["disp_local.png", "displacement.png"],
                    'shear': ["csforce_local.png", "area_shear.png"]
                }
                p = self._first_existing(ydir, name_map[t])
                img = self._read_rgb(p)
                key_map = {
                    'cnorm': ("cnforce_local", "CNORM"),
                    'stress1': ("nforce_local", "S11"),
                    'stress2': ("sforce_local", "S12"),
                    'disp': ("disp_local", "UU1"),
                    'shear': ("csforce_local", "CNAREA")
                }
                bk = self._bounds_for(bounds, *key_map[t])
                y3 = self._decode_3ch(img, bk) if bk else (img.astype(np.float32) / 255.0)
                if self.normalization and t == 'cnorm':
                    y3 -= FORCE_MEANS[None, None, :]
                if self.normalization and t == 'disp':
                    y3 -= DISP_MEANS[None, None, :]
                parts.append(y3)

        y_np = np.concatenate(parts, axis=2)

        if self.contiguous_on_direction:
            H, W, C = y_np.shape
            if 'depth' in self.output_types:
                depth = y_np[:, :, [0]]
                dirs  = y_np[:, :, 1:]
            else:
                depth = None
                dirs  = y_np
            dirs = dirs.reshape(H, W, -1, 3).transpose(0, 1, 3, 2).reshape(H, W, -1)
            y_np = np.concatenate([f for f in [depth, dirs] if f is not None], axis=2)

        return y_np

    # ---------- iteration plan ----------
    def _plan(self):
        items = list_sample_dirs(self.samples_roots)
        rnd = random.Random(self.rng_seed)
        items = list(items)
        rnd.shuffle(items)
        items = shard_for_worker_and_rank(items)
        items = BufferedShuffle(items, bufsize=self.shuffle_buffer, rng=rnd)
        if self.epoch_size is not None:
            items = islice(items, self.epoch_size)
        return items

    def __iter__(self):
        for _, xdir, ydir in self._plan():
            try:
                X = self._load_X(xdir)
                if self.is_mae:
                    yield X, torch.tensor(0, dtype=torch.long)
                    continue

                y_np = self._load_y_numpy(ydir)
                y = self.transform_y(y_np).float()

                if self.output_mask is not None:
                    mask = torch.from_numpy(self.output_mask) if isinstance(self.output_mask, np.ndarray) else self.output_mask
                    if mask.ndim == 2:
                        mask = mask.unsqueeze(0)
                    if mask.shape[-2:] != y.shape[-2:]:
                        mask = TF.resize(mask, y.shape[-2:], interpolation=transforms.InterpolationMode.BILINEAR)
                    y = y * mask.to(y.dtype)

                yield X, y
            except Exception:
                continue

    def __len__(self):
        """
        Returns number of samples THIS rank will process in an epoch.
        """
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if self.epoch_size is not None:
            return self.epoch_size // world_size
        return self._total_samples // world_size
