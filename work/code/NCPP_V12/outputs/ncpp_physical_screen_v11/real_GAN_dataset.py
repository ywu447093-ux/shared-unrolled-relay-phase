"""V11 rPMT透明细胞数据加载器；训练时在线D4增强。"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from amplitude_support import estimate_support_from_amplitude


class RelaySPSDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        total_samples: Optional[int] = None,
        recompute_support: bool = False,
        augment: bool = False,
    ) -> None:
        self.data_dir = os.path.abspath(data_dir)
        self.split = split
        self.recompute_support = bool(recompute_support)
        self.augment = bool(augment and split == "train")
        self.paths = {
            name: os.path.join(self.data_dir, name)
            for name in (
                "phase", "amp", "sps", "params", "amp_measurement",
                "amp_flat_field", "support_from_amp", "saturation",
            )
        }
        for name, path in self.paths.items():
            if not os.path.isdir(path):
                raise FileNotFoundError(f"缺少数据目录 {name}: {path}")

        # 明确拒绝旧的相位/标注 mask 数据接口，防止误用。
        legacy_mask = os.path.join(self.data_dir, "mask")
        if os.path.isdir(legacy_mask):
            raise RuntimeError(
                "检测到旧 mask 目录。V10 不读取该目录；请用 prepare_real_dataset.py "
                "重新生成到一个空的新目录。"
            )

        self.file_list = sorted(glob.glob(os.path.join(self.paths["sps"], f"{split}_*.npy")))
        if total_samples is not None:
            self.file_list = self.file_list[: int(total_samples)]
        if not self.file_list:
            raise FileNotFoundError(f"{self.paths['sps']} 中没有 {split}_*.npy")

        probe = np.load(self.file_list[0], mmap_mode="r")
        if probe.ndim != 2 or probe.shape[0] != probe.shape[1]:
            raise ValueError(f"SPS 必须为二维正方形，实际 {probe.shape}")
        self.image_size = int(probe.shape[0])

    @staticmethod
    def _normalize(value: float, low: float, high: float) -> float:
        return float(np.clip(2.0 * (value - low) / (high - low + 1e-12) - 1.0, -1.0, 1.0))

    @staticmethod
    def _load_2d(path: str, name: str) -> np.ndarray:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"缺少 {name}: {path}")
        array = np.load(path).astype(np.float32, copy=False)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError(f"{name} 必须是有限二维数组，实际 shape={array.shape}")
        return array

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        sps_path = self.file_list[index]
        name = Path(sps_path).stem
        sps = self._load_2d(sps_path, "sps")
        amplitude = self._load_2d(os.path.join(self.paths["amp"], f"{name}.npy"), "amp")
        phase = self._load_2d(os.path.join(self.paths["phase"], f"{name}.npy"), "phase")
        amp_measurement = self._load_2d(
            os.path.join(self.paths["amp_measurement"], f"{name}.npy"), "amp_measurement"
        )
        amp_flat_field = self._load_2d(
            os.path.join(self.paths["amp_flat_field"], f"{name}.npy"), "amp_flat_field"
        )
        saturation = self._load_2d(
            os.path.join(self.paths["saturation"], f"{name}.npy"), "saturation"
        )
        expected_shape = (self.image_size, self.image_size)
        for label, array in (
            ("amp", amplitude), ("phase", phase), ("amp_measurement", amp_measurement),
            ("amp_flat_field", amp_flat_field),
            ("saturation", saturation),
        ):
            if array.shape != expected_shape:
                raise ValueError(f"{name} 的 {label} 尺寸 {array.shape} 与 SPS {expected_shape} 不一致")

        if self.recompute_support:
            support = estimate_support_from_amplitude(
                amp_measurement, flat_field=amp_flat_field
            ).soft
        else:
            support = self._load_2d(
                os.path.join(self.paths["support_from_amp"], f"{name}.npy"),
                "support_from_amp",
            )
        if support.shape != expected_shape:
            raise ValueError(f"{name} 的 support 尺寸不一致。")

        if self.augment:
            variant = int(torch.randint(0, 8, ()).item())
            rotation = variant % 4
            do_flip = variant >= 4

            def transform(array: np.ndarray) -> np.ndarray:
                transformed = np.rot90(array, rotation)
                if do_flip:
                    transformed = np.fliplr(transformed)
                return np.ascontiguousarray(transformed)

            sps, amplitude, phase = transform(sps), transform(amplitude), transform(phase)
            amp_measurement, amp_flat_field = transform(amp_measurement), transform(amp_flat_field)
            support, saturation = transform(support), transform(saturation)

        params_path = os.path.join(self.paths["params"], f"{name}.txt")
        params = np.atleast_1d(np.loadtxt(params_path, dtype=np.float64))
        if params.size < 7 or not np.isfinite(params).all():
            raise ValueError(f"物理参数无效: {params_path}")
        if params[6] <= 0:
            raise ValueError(f"相机模量增益必须为正: {params_path}")
        wavelength, distance, l_s = map(float, params[:3])
        params_norm = torch.tensor(
            [
                self._normalize(wavelength, 525e-9, 540e-9),
                self._normalize(distance, 0.85, 1.15),
                self._normalize(np.log10(max(l_s, 1e-8)), -0.66, -0.45),
            ],
            dtype=torch.float32,
        )

        target = np.stack([amplitude, phase], axis=0)
        to_tensor = lambda x: torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0)
        return (
            to_tensor(sps),
            params_norm,
            torch.tensor(params, dtype=torch.float32),
            to_tensor(amp_measurement),
            to_tensor(amp_flat_field),
            to_tensor(np.clip(support, 0.0, 1.0)),
            to_tensor((saturation > 0.5).astype(np.float32)),
            torch.from_numpy(np.ascontiguousarray(target)).float(),
        )


# 保留旧名称，避免用户已有 import 立即失效。
MNISTSPSDataset = RelaySPSDataset


__all__ = ["RelaySPSDataset", "MNISTSPSDataset"]
