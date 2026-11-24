"""从可采集的振幅图像估计透明细胞软支持域。

这个模块不接收相位图、相位标签或标注分割图。训练和推理使用同一入口，
从而避免把待求相位的空间位置通过标签泄漏给 NCPP 展开层。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


EPS = 1e-6


@dataclass(frozen=True)
class SupportEstimate:
    soft: np.ndarray
    hard: np.ndarray
    background: np.ndarray
    absorption_score: np.ndarray
    illumination_mask: np.ndarray
    threshold: float
    contrast_snr: float
    valid: bool
    source: str = "amplitude_measurement_only"


def _float01(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("振幅图像为空。")
    image = np.asarray(image)
    if image.ndim == 3:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError(f"振幅图像必须是二维数组，实际 shape={image.shape}")
    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    max_value = float(image.max()) if image.size else 0.0
    if max_value > 1.5:
        image = image / (255.0 if max_value <= 255.0 else max_value)
    return np.clip(image, 0.0, 1.2)


def _odd_kernel(size: int, ratio: float, minimum: int = 9) -> int:
    kernel = max(minimum, int(round(size * ratio)))
    if kernel % 2 == 0:
        kernel += 1
    max_kernel = size if size % 2 == 1 else size - 1
    return max(3, min(kernel, max_kernel))


def _border_values(image: np.ndarray, ratio: float = 0.08) -> np.ndarray:
    h, w = image.shape
    border = max(2, int(round(min(h, w) * ratio)))
    return np.concatenate(
        [
            image[:border].ravel(),
            image[-border:].ravel(),
            image[:, :border].ravel(),
            image[:, -border:].ravel(),
        ]
    )


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= min_area:
            cleaned[labels == index] = 1
    return cleaned


def estimate_support_from_amplitude(
    amplitude: np.ndarray,
    flat_field: Optional[np.ndarray] = None,
    *,
    min_relative_absorption: float = 0.004,
    mad_scale: float = 3.5,
    min_area_ratio: float = 0.0015,
    background_kernel_ratio: float = 0.31,
    soft_edge_sigma: float = 1.25,
    illumination_floor_ratio: float = 0.12,
    detection_floor_ratio: float = 0.30,
    min_contrast_snr: float = 4.0,
) -> SupportEstimate:
    """估计弱吸收细胞的软支持域。

    ``flat_field`` 可来自无样本参考帧。若未提供，则用形态学闭运算估计缓慢
    变化的照明背景。检测失败时返回全 1 软支持域，让模型退化为无支持约束，
    而不是错误地删除弱细胞。
    """
    amp = _float01(amplitude)
    h, w = amp.shape
    if min(h, w) < 16:
        raise ValueError("振幅图像太小，无法稳定估计支持域。")

    smooth = cv2.GaussianBlur(amp, (0, 0), 0.65, borderType=cv2.BORDER_REFLECT101)
    if flat_field is not None:
        background = _float01(flat_field)
        if background.shape != amp.shape:
            raise ValueError("flat_field 与 amplitude 尺寸不一致。")
        background = cv2.GaussianBlur(background, (0, 0), 1.0)
        illumination_floor = illumination_floor_ratio * float(np.percentile(background, 99.0))
        illuminated = background >= max(illumination_floor, 0.01)
        # 独立拍摄的样本帧与平场帧在光阑陡峭边缘会有亚像素错位和较大的
        # 相对读出噪声。该区域可以参与成像，但不能被误判成一圈“细胞”。
        detection_floor = detection_floor_ratio * float(np.percentile(background, 99.0))
        detection_region = background >= max(detection_floor, 0.02)
    else:
        kernel_size = _odd_kernel(min(h, w), background_kernel_ratio)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        background = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, kernel)
        border_level = float(np.percentile(_border_values(smooth), 65.0))
        background = np.maximum(background, 0.985 * border_level)
        illuminated = np.ones_like(amp, dtype=bool)
        detection_region = illuminated

    background = np.clip(background, EPS, 1.2).astype(np.float32)
    absorption = np.clip((background - smooth) / background, 0.0, 1.0)
    absorption[~detection_region] = 0.0

    # 有限光阑会使图像边界变暗，不能再把边界当作噪声统计区。使用整个有效
    # 照明区的低吸收主体估计噪声，细胞只占少数时仍保持稳健。
    valid_score = absorption[detection_region]
    if valid_score.size < max(32, int(0.05 * h * w)):
        raise ValueError("有效照明区域过小，无法估计振幅支持域。")
    cutoff = float(np.percentile(valid_score, 70.0))
    background_score = valid_score[valid_score <= cutoff]
    median = float(np.median(background_score))
    mad = float(np.median(np.abs(background_score - median)))
    robust_sigma = max(1.4826 * mad, 5e-4)
    threshold = max(float(min_relative_absorption), median + mad_scale * robust_sigma)
    contrast_snr = float((np.percentile(absorption, 95.0) - median) / robust_sigma)

    raw = ((absorption >= threshold) & detection_region).astype(np.uint8)
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, small, iterations=1)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, small, iterations=2)
    min_area = max(8, int(round(h * w * min_area_ratio)))
    hard = _remove_small_components(raw, min_area)

    # 低 SNR 下二值位置比“无位置先验”更危险；回退到完整有效孔径。
    valid = bool(hard.sum() >= min_area and contrast_snr >= min_contrast_snr)
    if valid:
        hard = cv2.dilate(hard, small, iterations=2)
        distance_soft = cv2.GaussianBlur(
            hard.astype(np.float32), (0, 0), soft_edge_sigma,
            borderType=cv2.BORDER_REFLECT101,
        )
        transition = max(0.35 * threshold, 0.0015)
        score_soft = 1.0 / (1.0 + np.exp(-np.clip((absorption - threshold) / transition, -20.0, 20.0)))
        soft = np.maximum(distance_soft, score_soft * hard).astype(np.float32)
        soft = np.clip(soft, 0.0, 1.0)
    else:
        # 安全回退：错误的空/小支持域比不加约束更危险。
        # 回退为整个有效照明孔径，相当于不施加细胞位置先验。
        hard = illuminated.astype(np.uint8)
        soft = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), 1.0)

    return SupportEstimate(
        soft=soft,
        hard=hard.astype(np.float32),
        background=background,
        absorption_score=absorption.astype(np.float32),
        illumination_mask=illuminated.astype(np.float32),
        threshold=float(threshold),
        contrast_snr=contrast_snr,
        valid=valid,
    )


__all__ = ["SupportEstimate", "estimate_support_from_amplitude"]
