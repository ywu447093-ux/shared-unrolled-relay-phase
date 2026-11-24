"""生成无相位掩膜泄漏的 rPMT/NCPP 透明细胞数据集。

关键约定
1. 只读取 Annotated QPM 数据集中的 ``*_img.tif``，绝不把 ``*_mask.png``
   当成相位样本，也不把标注掩膜送入网络。
2. ``amp``/``phase`` 是干净监督标签；``amp_measurement`` 是独立的带噪振幅
   采集模拟。空间支持域仅由后者计算，临床推理使用同一接口。
3. SPS、标签和辅助测量均保存为 float32 .npy，避免 8 bit 量化弱高频。
4. 物屏距离和中继屏尺寸限定在论文米级显微实验范围；物面采样由
   ``dx_obj=lambda*Z/L_screen`` 推导，避免产生数米到十余米的虚假屏幕。
5. 有限照明孔径在计算边界衰减到零，避免周期常量场产生单像素 DC 假象。
"""

from __future__ import annotations

import csv
import glob
import json
import os
import argparse
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np
import tifffile

from amplitude_support import estimate_support_from_amplitude


EPS = 1e-12
FRESNEL_NUMBER_THRESHOLD = 0.01


def calibrate_sampling_interval(
    L_ref: float, N_ref: int, N: int, lambda_: float, Z_OS: float
) -> Tuple[float, float]:
    if min(L_ref, N_ref, N, lambda_, Z_OS) <= 0:
        raise ValueError("标定参数必须为正数。")
    dx_obs = L_ref / float(N_ref)
    dx_obj = lambda_ * Z_OS / (N * dx_obs)
    return dx_obj, dx_obs


def fresnel_number(side_length: float, wavelength: float, distance: float) -> float:
    """以半边长为特征孔径计算 Fresnel 数。"""
    return float((0.5 * side_length) ** 2 / (wavelength * distance + EPS))


def adaptive_propagation_sfft(
    u_in: np.ndarray,
    distance: float,
    wavelength: float,
    dx_in: float,
    fresnel_threshold: float = FRESNEL_NUMBER_THRESHOLD,
) -> Tuple[np.ndarray, float, float]:
    """单 FFT 传播；返回复场、输出采样间隔和 Fresnel 数。"""
    if u_in.ndim != 2 or u_in.shape[0] != u_in.shape[1]:
        raise ValueError("当前传播实现要求二维正方形输入。")
    if min(distance, wavelength, dx_in) <= 0:
        raise ValueError("distance、wavelength 和 dx_in 必须为正。")

    n = u_in.shape[0]
    k = 2.0 * np.pi / wavelength
    side_length = n * dx_in
    f_number = fresnel_number(side_length, wavelength, distance)
    use_fresnel = f_number >= fresnel_threshold

    coord = (np.arange(n, dtype=np.float64) - n // 2) * dx_in
    yy, xx = np.meshgrid(coord, coord, indexing="ij")
    phase_in = (
        np.exp(1j * k * (xx * xx + yy * yy) / (2.0 * distance))
        if use_fresnel else 1.0
    )
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_in * phase_in)))

    dx_out = wavelength * distance / (n * dx_in)
    if use_fresnel:
        out_coord = (np.arange(n, dtype=np.float64) - n // 2) * dx_out
        yy_out, xx_out = np.meshgrid(out_coord, out_coord, indexing="ij")
        phase_out = np.exp(1j * k * (xx_out * xx_out + yy_out * yy_out) / (2.0 * distance))
    else:
        phase_out = 1.0
    scaling = (dx_in * dx_in) / (1j * wavelength * distance)
    return (phase_out * spectrum * scaling).astype(np.complex64), float(dx_out), f_number


def _read_qpm(path: str) -> np.ndarray:
    try:
        image = tifffile.imread(path) if path.lower().endswith((".tif", ".tiff")) else cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except Exception as exc:
        raise RuntimeError(f"读取失败: {path}") from exc
    if image is None:
        raise RuntimeError(f"读取失败: {path}")
    image = np.asarray(image)
    if image.ndim == 3:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError(f"仅支持二维 QPM，实际 shape={image.shape}")
    return np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _pad_to_size(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape
    pad_h, pad_w = max(0, size - h), max(0, size - w)
    if pad_h == 0 and pad_w == 0:
        return image
    top, left = pad_h // 2, pad_w // 2
    return cv2.copyMakeBorder(
        image, top, pad_h - top, left, pad_w - left, cv2.BORDER_REFLECT_101
    )


def content_aware_native_crop(
    image: np.ndarray, size: int, rng: np.random.Generator, candidates: int = 16
) -> np.ndarray:
    """在原生网格上选含结构的 crop；不读取标注掩膜。"""
    image = _pad_to_size(image, size)
    h, w = image.shape
    if (h, w) == (size, size):
        return image.copy()
    positions = [((h - size) // 2, (w - size) // 2)]
    positions.extend(
        (int(rng.integers(0, h - size + 1)), int(rng.integers(0, w - size + 1)))
        for _ in range(max(1, candidates - 1))
    )
    best, best_score = None, -np.inf
    for y0, x0 in positions:
        crop = image[y0:y0 + size, x0:x0 + size]
        gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
        score = float(np.mean(np.hypot(gx, gy)) + 0.2 * np.std(crop))
        if score > best_score:
            best, best_score = crop, score
    if best is None:
        raise RuntimeError("无法选择有效 QPM crop。")
    return best.astype(np.float32, copy=True)


def qpm_to_phase_fraction(crop: np.ndarray) -> np.ndarray:
    """扣除边界背景，将 QPM 正光程信号稳健映射到 [0,1]。"""
    border = max(3, min(crop.shape) // 12)
    border_values = np.concatenate(
        [crop[:border].ravel(), crop[-border:].ravel(), crop[:, :border].ravel(), crop[:, -border:].ravel()]
    )
    corrected = crop - float(np.median(border_values))
    positive = np.clip(corrected, 0.0, None)
    scale = float(np.percentile(positive, 99.5))
    if not np.isfinite(scale) or scale <= 1e-7:
        raise ValueError("QPM crop 中没有足够的正相位信号。")
    return np.clip(positive / scale, 0.0, 1.0).astype(np.float32)


def generate_qpm_proxy(
    phase_fraction: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = phase_fraction.shape
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    # 有限超高斯照明孔径：中心近似平顶，边界连续衰减至零。真实rPMT的扩束
    # 光斑尺寸有限，不能用跨边界的常量场做周期 FFT。
    aperture = np.exp(-np.power(radius / 0.86, 12.0))
    broad_beam = (0.90 + 0.10 * np.exp(-(radius / 0.72) ** 2)) * aperture

    coarse = rng.normal(0.0, 1.0, (max(4, h // 16), max(4, w // 16))).astype(np.float32)
    coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    coarse /= float(np.max(np.abs(coarse)) + EPS)
    illumination = np.clip(broad_beam * (1.0 + 0.012 * coarse), 0.0, 1.05)

    absorption_strength = float(rng.uniform(0.025, 0.065))
    absorption_texture = cv2.GaussianBlur(phase_fraction, (0, 0), float(rng.uniform(0.35, 0.9)))
    amplitude = illumination * (1.0 - absorption_strength * absorption_texture)
    amplitude = np.clip(amplitude, 0.0, 1.1).astype(np.float32)

    # 透明弱相位样本限定在主值区间内。超过 pi 的标签与单帧强度观测在
    # 2*pi 意义下等价，却会被普通 L1/SSIM 当成完全不同的回归目标。
    phase_scale = float(rng.uniform(0.45 * np.pi, 0.95 * np.pi))
    phase = (phase_fraction * phase_scale * aperture).astype(np.float32)
    return amplitude, phase, illumination.astype(np.float32)


def simulate_amplitude_measurement(
    clean_amplitude: np.ndarray,
    rng: np.random.Generator,
    gain: float | None = None,
) -> np.ndarray:
    """生成与干净振幅标签独立的临床可采集辅助振幅帧。"""
    sigma = float(rng.uniform(0.35, 0.9))
    blurred = cv2.GaussianBlur(clean_amplitude, (0, 0), sigma, borderType=cv2.BORDER_REFLECT101)
    intensity = np.clip(blurred, 0.0, None) ** 2
    full_well = float(rng.uniform(18000.0, 36000.0))
    gain = float(rng.uniform(0.92, 1.02) if gain is None else gain)
    expected = np.clip(intensity * gain * full_well, 0.0, None)
    electrons = rng.poisson(expected).astype(np.float32)
    electrons += rng.normal(0.0, rng.uniform(1.0, 3.0), intensity.shape).astype(np.float32)
    measured = np.sqrt(np.clip(electrons / full_well, 0.0, None))
    return np.clip(measured, 0.0, 1.2).astype(np.float32)


def _simulate_exposure(
    intensity: np.ndarray,
    intensity_peak: float,
    exposure: float,
    full_well: float,
    read_noise: float,
    bias: float,
    rng: np.random.Generator,
) -> np.ndarray:
    expected = intensity / max(intensity_peak, EPS) * (full_well - bias) * exposure
    counts = rng.poisson(np.clip(expected, 0.0, None)).astype(np.float32)
    counts += rng.normal(bias, read_noise, intensity.shape).astype(np.float32)
    return np.clip(counts, 0.0, full_well)


def generate_real_sps_corrected(
    phase: np.ndarray, amplitude: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    n = phase.shape[0]
    if phase.shape != amplitude.shape or phase.ndim != 2 or phase.shape[1] != n:
        raise ValueError("phase 与 amplitude 必须是同尺寸正方形二维数组。")

    # 论文的米级显微场景：约 1 m 物屏距离、约 280 mm 有效中继屏。
    wavelength = float(rng.uniform(527e-9, 537e-9))
    distance = float(rng.uniform(0.90, 1.10))
    screen_side = float(rng.uniform(0.25, 0.32))
    dx_out_target = screen_side / n
    dx_in = wavelength * distance / (n * dx_out_target)
    field = amplitude * np.exp(1j * phase)
    screen, dx_out, f_number = adaptive_propagation_sfft(field, distance, wavelength, dx_in)
    intensity = np.abs(screen) ** 2

    # Lambertian屏幕的慢变反射率与相机对屏幕的有限分辨率。漫反射不应把SPS
    # 任意摊平，只引入局部反射率和成像模糊。
    coarse = rng.normal(0.0, 1.0, (8, 8)).astype(np.float32)
    coarse = cv2.resize(coarse, (n, n), interpolation=cv2.INTER_CUBIC)
    coarse /= float(np.max(np.abs(coarse)) + EPS)
    screen_reflectance = np.clip(0.96 + 0.025 * coarse, 0.90, 1.02)
    intensity = intensity * screen_reflectance
    intensity = cv2.GaussianBlur(
        intensity.astype(np.float32), (0, 0), float(rng.uniform(0.25, 0.55)),
        borderType=cv2.BORDER_REFLECT101,
    )

    full_well = float(rng.uniform(24000.0, 42000.0))
    read_noise = float(rng.uniform(1.5, 3.5))
    bias = float(rng.uniform(80.0, 180.0))
    exposure = float(rng.uniform(1.25, 1.75))
    intensity_peak = float(np.percentile(intensity, 99.9)) + EPS
    camera_modulus_gain = float(
        np.sqrt((full_well - bias) * exposure / intensity_peak)
    )
    captured = _simulate_exposure(
        intensity, intensity_peak, exposure, full_well, read_noise, bias, rng
    )
    dark = np.clip(rng.normal(bias, read_noise, intensity.shape), 0.0, full_well).astype(np.float32)
    background = float(np.median(dark))

    # 单次曝光：中央零级允许真实截断。饱和值在NCPP中只作为模量下界，
    # 不再用第二次短曝光恢复后又错误地标记为饱和。
    saturated = captured >= 0.985 * full_well
    modulus = np.sqrt(np.clip(captured - background, 0.0, None) + EPS).astype(np.float32)
    s_min, s_max = float(modulus.min()), float(modulus.max())
    if s_max <= s_min + EPS:
        raise ValueError("SPS 动态范围过小。")
    sps = ((modulus - s_min) / (s_max - s_min)).astype(np.float32)
    l_s = float(dx_out * n)
    if abs(l_s - screen_side) / screen_side > 1e-5:
        raise RuntimeError("传播后的屏幕采样与目标物理尺寸不一致。")

    energy = np.maximum(sps.astype(np.float64), 0.0) ** 2
    yy, xx = np.mgrid[:n, :n]
    total = float(energy.sum() + EPS)
    cy = float((yy * energy).sum() / total)
    cx = float((xx * energy).sum() / total)
    radius = np.hypot(xx - cx, yy - cy)
    order = np.argsort(radius.ravel())
    cdf = np.cumsum(energy.ravel()[order]) / total
    sorted_radius = radius.ravel()[order]
    r50 = float(sorted_radius[np.searchsorted(cdf, 0.50)])
    r90 = float(sorted_radius[np.searchsorted(cdf, 0.90)])
    area_10 = float(np.mean(sps >= 0.10 * float(sps.max())))
    # 第 7 项把电子计数模量还原为传播光场模量；没有该标定时，
    # 数据一致性层会在相差数个数量级的单位之间比较。
    params = np.array(
        [wavelength, distance, l_s, s_min, s_max, f_number, camera_modulus_gain],
        dtype=np.float64,
    )
    diagnostics = {
        "sps_r50_pixels": r50,
        "sps_r90_pixels": r90,
        "sps_area_above_10pct_peak": area_10,
        "object_pixel_size_m": float(dx_in),
        "object_side_length_m": float(n * dx_in),
        "screen_pixel_size_m": float(dx_out),
        "screen_side_length_m": l_s,
    }
    return sps, saturated.astype(np.float32), params, diagnostics


def _d4(image: np.ndarray) -> Iterable[np.ndarray]:
    for k in range(4):
        rotated = np.rot90(image, k)
        yield np.ascontiguousarray(rotated)
        yield np.ascontiguousarray(np.fliplr(rotated))


def _find_qpm_images(raw_dir: str) -> list[str]:
    preferred = sorted(glob.glob(os.path.join(raw_dir, "**", "*_img.tif"), recursive=True))
    if preferred:
        return preferred
    candidates = glob.glob(os.path.join(raw_dir, "**", "*.*"), recursive=True)
    excluded = ("mask", "support", "label", "seg")
    return sorted(
        p for p in candidates
        if p.lower().endswith((".tif", ".tiff"))
        and not any(token in os.path.basename(p).lower() for token in excluded)
    )


def _stratified_split(
    paths: Sequence[str], ratio: float, rng: np.random.Generator
) -> Tuple[list[str], list[str]]:
    groups: dict[str, list[str]] = {}
    for path in paths:
        groups.setdefault(Path(path).parent.name, []).append(path)
    train, test = [], []
    for group in sorted(groups):
        items = list(groups[group])
        rng.shuffle(items)
        if len(items) == 1:
            train.extend(items)
            continue
        cut = min(max(int(round(len(items) * ratio)), 1), len(items) - 1)
        train.extend(items[:cut])
        test.extend(items[cut:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _save_sample(
    root: str,
    name: str,
    amplitude: np.ndarray,
    phase: np.ndarray,
    sps: np.ndarray,
    amplitude_measurement: np.ndarray,
    amplitude_flat_field: np.ndarray,
    support: np.ndarray,
    saturation: np.ndarray,
    params: np.ndarray,
    support_meta: dict,
) -> None:
    arrays = {
        "amp": amplitude,
        "phase": phase,
        "sps": sps,
        "amp_measurement": amplitude_measurement,
        "amp_flat_field": amplitude_flat_field,
        "support_from_amp": support,
        "saturation": saturation,
    }
    for folder, array in arrays.items():
        np.save(os.path.join(root, folder, f"{name}.npy"), array.astype(np.float32))
    np.savetxt(os.path.join(root, "params", f"{name}.txt"), params, fmt="%.12e")
    with open(os.path.join(root, "support_meta", f"{name}.json"), "w", encoding="utf-8") as stream:
        json.dump(support_meta, stream, ensure_ascii=False, indent=2)


def generate_real_dataset(
    raw_data_dir: str,
    save_dir: str,
    img_size: int = 128,
    split_ratio: float = 0.8,
    seed: int = 2026,
    simulations_per_image: int = 2,
) -> None:
    if img_size < 32 or img_size % 4:
        raise ValueError("img_size 必须不小于 32 且能被 4 整除。")
    if not 0.0 < split_ratio < 1.0:
        raise ValueError("split_ratio 必须位于 (0,1)。")
    if simulations_per_image < 1:
        raise ValueError("simulations_per_image 必须大于 0。")
    save_path = Path(save_dir)
    if save_path.exists() and any(save_path.iterdir()):
        raise FileExistsError(
            f"输出目录必须为空，防止旧 mask 或旧划分混入新数据集: {save_path}"
        )

    folders = (
        "amp", "phase", "sps", "amp_measurement", "amp_flat_field", "support_from_amp",
        "saturation", "params", "support_meta",
    )
    for folder in folders:
        os.makedirs(os.path.join(save_dir, folder), exist_ok=True)

    paths = _find_qpm_images(raw_data_dir)
    if not paths:
        raise FileNotFoundError(f"在 {raw_data_dir} 未找到 QPM *_img.tif。")
    rng = np.random.default_rng(seed)
    train_paths, test_paths = _stratified_split(paths, split_ratio, rng)
    print(f"仅使用 {len(paths)} 张 QPM *_img.tif；标注 mask 不参与训练或约束。")
    print(f"训练原图 {len(train_paths)}，验证原图 {len(test_paths)}，输出 {img_size}×{img_size}。")

    records: list[dict[str, str]] = []

    def process(items: list[str], split: str) -> int:
        index = 0
        for source_index, source in enumerate(items):
            try:
                crop = content_aware_native_crop(_read_qpm(source), img_size, rng)
                phase_fraction = qpm_to_phase_fraction(crop)
            except Exception as exc:
                print(f"[跳过] {source}: {exc}")
                continue
            for simulation in range(simulations_per_image):
                amplitude, phase, flat_field = generate_qpm_proxy(phase_fraction, rng)
                amplitude_gain = float(rng.uniform(0.92, 1.02))
                amp_measurement = simulate_amplitude_measurement(amplitude, rng, gain=amplitude_gain)
                amp_flat_field = simulate_amplitude_measurement(flat_field, rng, gain=amplitude_gain)
                support_result = estimate_support_from_amplitude(
                    amp_measurement, flat_field=amp_flat_field
                )
                sps, saturation, params, sps_diagnostics = generate_real_sps_corrected(
                    phase, amplitude, rng
                )
                name = f"{split}_{index:06d}"
                meta = {
                    "source": support_result.source,
                    "valid": support_result.valid,
                    "threshold": support_result.threshold,
                    "contrast_snr": support_result.contrast_snr,
                    "uses_phase": False,
                    "uses_annotation_mask": False,
                    "illumination_fraction": float(support_result.illumination_mask.mean()),
                    **sps_diagnostics,
                }
                _save_sample(
                    save_dir, name, amplitude, phase, sps, amp_measurement, amp_flat_field,
                    support_result.soft, saturation, params, meta,
                )
                records.append(
                    {
                        "name": name,
                        "split": split,
                        "source_qpm": source,
                        "cell_type": Path(source).parent.name,
                        "simulation_index": str(simulation),
                        "support_source": support_result.source,
                        "support_valid": str(support_result.valid),
                    }
                )
                index += 1
            if (source_index + 1) % 25 == 0 or source_index + 1 == len(items):
                print(f"[{split.upper()}] {source_index + 1}/{len(items)} 原图，{index} 样本")
        return index

    n_train = process(train_paths, "train")
    n_test = process(test_paths, "test")
    if not records:
        raise RuntimeError("没有生成任何有效样本。")
    manifest_path = os.path.join(save_dir, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"完成：训练 {n_train}，验证 {n_test}。支持域来源固定为 amplitude_measurement_only。")


def main() -> None:
    base_path = r"C:\Users\Lenovo\Desktop\Work\computing_imaging\GAN_SPS"
    parser = argparse.ArgumentParser(description="生成物理屏幕标定的 V11 数据集。")
    parser.add_argument("--raw-data-dir", default=os.path.join(base_path, "raw_cells"))
    parser.add_argument("--save-dir", default=os.path.join(base_path, "real_cell_dataset_v11_physical_screen"))
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--simulations-per-image", type=int, default=2)
    args = parser.parse_args()
    generate_real_dataset(
        raw_data_dir=args.raw_data_dir,
        save_dir=args.save_dir,
        img_size=args.img_size,
        split_ratio=args.split_ratio,
        seed=args.seed,
        simulations_per_image=args.simulations_per_image,
    )


if __name__ == "__main__":
    main()
