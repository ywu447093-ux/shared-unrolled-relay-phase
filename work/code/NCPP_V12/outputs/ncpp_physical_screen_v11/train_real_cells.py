"""训练V11物理屏幕标定版可微NCPP；评价不使用真值piston对齐。"""

from __future__ import annotations

import json
import math
import os
import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from real_GAN_dataset import RelaySPSDataset
from real_model import DeterministicPhysicalLoss, PhysicsUnrolledGenerator, _ssim_map


EPS = 1e-8


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / (weight.sum() + EPS)


def _charbonnier_map(value: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(value.square() + epsilon * epsilon)


METRIC_NAMES = (
    "amp_psnr",
    "amp_input_ssim",
    "amp_pred_ssim",
    "amp_ssim_gain",
    "phase_psnr",
    "phase_full_ssim",
    "phase_foreground_ssim",
    "phase_gradient_correlation",
    "phase_background_mae_rad",
    "phase_contrast_ratio",
)


def illumination_mask(flat_field: torch.Tensor) -> torch.Tensor:
    peak = flat_field.amax(dim=(-2, -1), keepdim=True).clamp_min(EPS)
    return (flat_field >= 0.12 * peak).to(flat_field.dtype)


def anchor_phase_to_amplitude_background(
    pred_phase: torch.Tensor,
    support: torch.Tensor,
    flat_field: torch.Tensor,
) -> torch.Tensor:
    """只用振幅支持域外的有效照明区校零；不读取真实相位。"""
    background = illumination_mask(flat_field) * (1.0 - torch.clamp(support, 0.0, 1.0))
    offset = (pred_phase * background).sum(dim=(-2, -1), keepdim=True) / (
        background.sum(dim=(-2, -1), keepdim=True) + EPS
    )
    return torch.clamp(pred_phase - offset, 0.0, math.pi)


def gradient_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    dx = (pred[..., 1:] - pred[..., :-1]) - (target[..., 1:] - target[..., :-1])
    dy = (pred[..., 1:, :] - pred[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])
    wx = 1.0 + 3.0 * 0.5 * (support[..., 1:] + support[..., :-1])
    wy = 1.0 + 3.0 * 0.5 * (support[..., 1:, :] + support[..., :-1, :])
    return _weighted_mean(_charbonnier_map(dx), wx) + _weighted_mean(_charbonnier_map(dy), wy)


def gradient_correlation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """整幅标签上的可微梯度相关损失；不构造相位前景掩膜。"""
    dx_p = pred[..., 1:] - pred[..., :-1]
    dx_t = target[..., 1:] - target[..., :-1]
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    pred_grad = torch.cat([dx_p.flatten(1), dy_p.flatten(1)], dim=1)
    true_grad = torch.cat([dx_t.flatten(1), dy_t.flatten(1)], dim=1)
    pred_grad = pred_grad - pred_grad.mean(dim=1, keepdim=True)
    true_grad = true_grad - true_grad.mean(dim=1, keepdim=True)
    numerator = (pred_grad * true_grad).sum(dim=1)
    denominator = (
        torch.linalg.vector_norm(pred_grad, dim=1)
        * torch.linalg.vector_norm(true_grad, dim=1)
        + EPS
    )
    return (1.0 - numerator / denominator).mean()


def laplacian_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    kernel = pred.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    kernel = kernel.view(1, 1, 3, 3)
    residual = F.conv2d(pred - target, kernel, padding=1)
    return _weighted_mean(_charbonnier_map(residual), 1.0 + 3.0 * support)


def _weighted_std(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    mean = (value * weight).sum(dim=(-2, -1), keepdim=True) / (
        weight.sum(dim=(-2, -1), keepdim=True) + EPS
    )
    variance = ((value - mean).square() * weight).sum(dim=(-2, -1), keepdim=True) / (
        weight.sum(dim=(-2, -1), keepdim=True) + EPS
    )
    return torch.sqrt(variance + EPS)


def mixed_loss_unrolled(
    pred: torch.Tensor,
    target: torch.Tensor,
    amplitude_measurement: torch.Tensor,
    amplitude_flat_field: torch.Tensor,
    support: torch.Tensor,
    physical_loss: torch.Tensor,
    detail_weight: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    amp_p, phase_p = pred[:, 0:1], pred[:, 1:2]
    amp_t, phase_t = target[:, 0:1], target[:, 1:2]
    support = torch.clamp(support, 0.0, 1.0)
    phase_p_n, phase_t_n = phase_p / math.pi, phase_t / math.pi
    # 训练标签是允许使用的监督信号，但不能从它生成模型的空间掩膜。振幅
    # 支持可能漏掉弱吸收细胞，因此这里只做轻量加权，主体损失覆盖整幅相位。
    spatial_weight = 1.0 + 0.75 * support
    phase_ssim_map = _ssim_map(phase_p_n, phase_t_n, data_range=1.0)
    illuminated = illumination_mask(amplitude_flat_field)
    outside_weight = illuminated * (1.0 - support)
    contrast_error = torch.abs(
        _weighted_std(phase_p_n, support) - _weighted_std(phase_t_n, support)
    ).mean()

    terms = {
        "phase_charb": 10.0 * _weighted_mean(
            _charbonnier_map(phase_p_n - phase_t_n), spatial_weight
        ),
        "phase_ssim_full": 6.0 * (1.0 - phase_ssim_map.mean()),
        "phase_ssim_support": 4.0 * (1.0 - _weighted_mean(phase_ssim_map, support)),
        "phase_contrast": 1.0 * contrast_error,
        "amp_charb": 2.0 * _charbonnier_map(amp_p - amp_t).mean(),
        "amp_ssim": 2.0 * (1.0 - _ssim_map(amp_p, amp_t, data_range=1.0).mean()),
        # 辅助振幅是带噪观测，只作弱约束，干净 amp 仍是监督标签。
        "amp_measurement": 0.25 * _charbonnier_map(amp_p - amplitude_measurement).mean(),
        "gradient": 10.0 * detail_weight * gradient_loss(phase_p_n, phase_t_n, support),
        "gradient_correlation": 2.0 * detail_weight * gradient_correlation_loss(
            phase_p_n, phase_t_n
        ),
        "laplacian": 4.0 * detail_weight * laplacian_loss(phase_p_n, phase_t_n, support),
        # 支持外不等于确定背景：弱吸收细胞可被振幅检测漏掉。仅保留轻量
        # 稳定项，避免该正则与真实但振幅不明显的相位区域直接冲突。
        "outside_phase": 0.05 * _weighted_mean(torch.abs(phase_p_n), outside_weight),
        "physics": 1.0 * physical_loss,
    }
    return sum(terms.values()), terms


def stage_weights(num_stages: int) -> list[float]:
    if num_stages < 1:
        raise ValueError("num_stages 必须大于 0。")
    progress = np.linspace(0.0, 1.0, num_stages, dtype=np.float32)
    weights = 0.03 + 0.97 * progress ** 3
    weights[-1] = 1.0
    return weights.tolist()


def compute_validation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
    amplitude_measurement: torch.Tensor,
    amplitude_flat_field: torch.Tensor,
) -> np.ndarray:
    """返回不依赖真值对齐的严格指标；真值前景只用于离线评价。"""
    rows = []
    for index in range(pred.shape[0]):
        amp_p = pred[index:index + 1, 0:1]
        amp_t = target[index:index + 1, 0:1]
        phase_p = pred[index:index + 1, 1:2]
        phase_t = target[index:index + 1, 1:2]
        amp_input = amplitude_measurement[index:index + 1]
        flat = amplitude_flat_field[index:index + 1]
        phase_range = math.pi
        threshold = torch.clamp(0.05 * phase_t.amax(), min=0.03)
        foreground = (phase_t > threshold).to(phase_t.dtype)
        illuminated = illumination_mask(flat)
        background = illuminated * (1.0 - foreground)

        mse_amp = F.mse_loss(amp_p, amp_t).item()
        mse_phase = F.mse_loss(phase_p, phase_t).item()
        amp_psnr = 10.0 * math.log10(1.0 / (mse_amp + 1e-10))
        phase_psnr = 10.0 * math.log10(phase_range ** 2 / (mse_phase + 1e-10))
        amp_input_ssim = float(_ssim_map(amp_input, amp_t, data_range=1.0).mean().item())
        amp_ssim = float(_ssim_map(amp_p, amp_t, data_range=1.0).mean().item())
        phase_map = _ssim_map(phase_p, phase_t, data_range=phase_range)
        phase_ssim = float(phase_map.mean().item())
        phase_foreground_ssim = float(_weighted_mean(phase_map, foreground).item())

        dx_p, dx_t = phase_p[..., 1:] - phase_p[..., :-1], phase_t[..., 1:] - phase_t[..., :-1]
        dy_p, dy_t = phase_p[..., 1:, :] - phase_p[..., :-1, :], phase_t[..., 1:, :] - phase_t[..., :-1, :]
        wx = 0.5 * (foreground[..., 1:] + foreground[..., :-1])
        wy = 0.5 * (foreground[..., 1:, :] + foreground[..., :-1, :])
        pred_grad = torch.cat([(dx_p * wx).flatten(), (dy_p * wy).flatten()])
        true_grad = torch.cat([(dx_t * wx).flatten(), (dy_t * wy).flatten()])
        pred_grad = pred_grad - pred_grad.mean()
        true_grad = true_grad - true_grad.mean()
        gradient_corr = float(
            (pred_grad * true_grad).sum()
            / (torch.linalg.vector_norm(pred_grad) * torch.linalg.vector_norm(true_grad) + EPS)
        )
        background_mae = float(_weighted_mean(torch.abs(phase_p), background).item())
        contrast_ratio = float(
            (_weighted_std(phase_p, foreground) / (_weighted_std(phase_t, foreground) + EPS)).item()
        )
        rows.append(
            (
                amp_psnr,
                amp_input_ssim,
                amp_ssim,
                amp_ssim - amp_input_ssim,
                phase_psnr,
                phase_ssim,
                phase_foreground_ssim,
                gradient_corr,
                background_mae,
                contrast_ratio,
            )
        )
    return np.asarray(rows, dtype=np.float64).mean(axis=0)


def _to_device(batch: Iterable[torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, ...]:
    return tuple(item.to(device, non_blocking=True) for item in batch)


def metrics_dict(metrics: np.ndarray) -> Dict[str, float]:
    return {name: float(value) for name, value in zip(METRIC_NAMES, metrics)}


def acceptance_status(
    metrics: np.ndarray,
    required_ssim: float,
    required_gradient_corr: float = 0.75,
    max_background_mae: float = 0.05,
) -> Dict[str, float | bool]:
    values = metrics_dict(metrics)
    contrast_ok = 0.75 <= values["phase_contrast_ratio"] <= 1.25
    threshold_met = bool(
        values["amp_pred_ssim"] >= required_ssim
        and values["amp_ssim_gain"] >= -0.003
        and values["phase_foreground_ssim"] >= required_ssim
        and values["phase_gradient_correlation"] >= required_gradient_corr
        and values["phase_background_mae_rad"] <= max_background_mae
        and contrast_ok
    )
    selection_score = (
        values["phase_foreground_ssim"]
        + 0.20 * values["phase_gradient_correlation"]
        + 0.05 * values["phase_full_ssim"]
        - 0.20 * values["phase_background_mae_rad"]
        - 0.05 * abs(math.log(max(values["phase_contrast_ratio"], 1e-4)))
    )
    return {
        "threshold_met": threshold_met,
        "selection_score": float(selection_score),
        "required_ssim": float(required_ssim),
        "required_gradient_correlation": float(required_gradient_corr),
        "max_background_mae_rad": float(max_background_mae),
        "contrast_ratio_ok": bool(contrast_ok),
    }


def save_checkpoint(
    path: str,
    model: PhysicsUnrolledGenerator,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: np.ndarray,
    required_ssim: float,
    required_gradient_corr: float,
    max_background_mae: float,
    img_size: int,
    num_stages: int,
    shared_prior: bool,
    features: int,
    seed: int,
    class_balanced_training: bool,
) -> None:
    values = metrics_dict(metrics)
    acceptance = acceptance_status(
        metrics, required_ssim, required_gradient_corr, max_background_mae
    )
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "img_size": int(img_size),
            "num_stages": int(num_stages),
            "shared_prior": bool(shared_prior),
            "features": int(features),
            "seed": int(seed),
            "class_balanced_training": bool(class_balanced_training),
            **values,
            **acceptance,
            "support_source": "amplitude_measurement_only",
            "uses_phase_mask": False,
            "ncpp_gamma_schedule": [1.2, 0.8],
            "phase_mask_used_for_evaluation_only": True,
            "phase_alignment": "none",
            "version": "V11-physical-screen-strict-metrics",
        },
        path,
    )


def train_real_cells(
    data_dir: str,
    save_dir: str,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-4,
    num_stages: int = 6,
    num_workers: int = 2,
    shared_prior: bool = True,
    required_ssim: float = 0.85,
    features: int = 48,
    seed: int = 2026,
    required_gradient_corr: float = 0.75,
    max_background_mae: float = 0.05,
    quiet: bool = False,
    init_checkpoint: str | None = None,
    class_balanced: bool = True,
    a2058_boost: float = 1.0,
    save_every_epoch: bool = False,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
    print("=" * 68)
    print("rPMT/NCPP V11：物理屏幕标定；无真值piston对齐；无相位mask输入")
    print(f"device={device} | stages={num_stages} | required SSIM={required_ssim:.2f}")
    print("NCPP gamma: 1.2 -> 0.8；支持域外振幅回归照明背景，不清零总场")
    print("=" * 68)

    train_set = RelaySPSDataset(data_dir, "train", augment=True)
    val_set = RelaySPSDataset(data_dir, "test")
    if train_set.image_size != val_set.image_size:
        raise ValueError("训练/验证图像尺寸不一致。")
    img_size = train_set.image_size
    pin = device.type == "cuda"
    sampler = None
    if class_balanced:
        manifest_path = Path(data_dir, "manifest.csv")
        with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
            manifest = {row["name"]: row for row in csv.DictReader(stream)}
        cell_types = [
            manifest[Path(path).stem]["cell_type"] for path in train_set.file_list
        ]
        counts = Counter(cell_types)
        weights_per_sample = torch.as_tensor(
            [
                (a2058_boost if cell_type == "A2058" else 1.0) / counts[cell_type]
                for cell_type in cell_types
            ],
            dtype=torch.double,
        )
        sampler_generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            weights_per_sample,
            num_samples=len(train_set),
            replacement=True,
            generator=sampler_generator,
        )
        print(
            f"启用细胞类型均衡采样：{dict(sorted(counts.items()))}；"
            f"A2058 boost={a2058_boost:.2f}"
        )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
        num_workers=num_workers, pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin,
    )

    model = PhysicsUnrolledGenerator(
        num_stages=num_stages,
        features=features,
        img_size=img_size,
        shared_prior=shared_prior,
    ).to(device)
    if init_checkpoint:
        checkpoint = torch.load(init_checkpoint, map_location=device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        incompatible = model.load_state_dict(state_dict, strict=False)
        allowed_missing = ("output_refiner.", "refine_scale_logit")
        illegal_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed_missing)
        ]
        if illegal_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "初始权重与模型不兼容："
                f"missing={illegal_missing}, unexpected={incompatible.unexpected_keys}"
            )
        print(f"已载入细节微调初始权重：{init_checkpoint}")
    physical_loss = DeterministicPhysicalLoss().to(device)
    refiner_parameters = list(model.output_refiner.parameters()) + [model.refine_scale_logit]
    refiner_ids = {id(parameter) for parameter in refiner_parameters}
    base_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in refiner_ids
    ]
    optimizer = optim.AdamW(
        [
            {"params": base_parameters, "lr": lr},
            {"params": refiner_parameters, "lr": 3.0 * lr},
        ],
        weight_decay=1e-5,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-7)
    weights = stage_weights(num_stages)

    best_joint = -float("inf")
    best_metrics = None
    history = []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        progress_bar = tqdm(
            train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]", disable=quiet
        )
        for batch in progress_bar:
            (
                sps, params_norm, params_phys, amp_meas, amp_flat,
                support, saturation, target,
            ) = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(
                sps, params_norm, params_phys,
                amplitude_measurement=amp_meas,
                amplitude_flat_field=amp_flat,
                support_mask=support,
                saturation_mask=saturation,
                device=device,
            )
            if not isinstance(predictions, list) or len(predictions) != num_stages:
                raise RuntimeError("训练模式应返回全部展开阶段。")

            total = sps.new_tensor(0.0)
            last_terms: Dict[str, torch.Tensor] = {}
            for stage_index, prediction in enumerate(predictions):
                progress = 1.0 if num_stages <= 1 else stage_index / float(num_stages - 1)
                p_loss = physical_loss(
                    prediction[:, 0:1], prediction[:, 1:2], sps, params_phys,
                    saturation_mask=saturation, device=device, stage_progress=progress,
                )
                loss, terms = mixed_loss_unrolled(
                    prediction, target, amp_meas, amp_flat, support, p_loss,
                    detail_weight=0.2 + 0.8 * progress ** 2,
                )
                total = total + weights[stage_index] * loss
                last_terms = terms
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += float(total.item())
            progress_bar.set_postfix(
                Loss=f"{total.item():.3f}",
                Phys=f"{last_terms['physics'].item():.3f}",
                ROI=f"{last_terms['phase_ssim_support'].item():.3f}",
            )
        scheduler.step()

        model.eval()
        metric_sum = np.zeros(len(METRIC_NAMES), dtype=np.float64)
        sample_count = 0
        with torch.no_grad():
            for batch in val_loader:
                (
                    sps, params_norm, params_phys, amp_meas, amp_flat,
                    support, saturation, target,
                ) = _to_device(batch, device)
                prediction = model(
                    sps, params_norm, params_phys,
                    amplitude_measurement=amp_meas,
                    amplitude_flat_field=amp_flat,
                    support_mask=support,
                    saturation_mask=saturation,
                    device=device,
                )
                batch_metrics = compute_validation_metrics(
                    prediction, target, support, amp_meas, amp_flat
                )
                batch_size_actual = sps.shape[0]
                metric_sum += batch_size_actual * batch_metrics
                sample_count += batch_size_actual
        if sample_count == 0:
            raise RuntimeError("验证集为空。")
        metrics = metric_sum / sample_count
        status = acceptance_status(
            metrics, required_ssim, required_gradient_corr, max_background_mae
        )
        joint = float(status["selection_score"])
        threshold_met = bool(status["threshold_met"])
        values = metrics_dict(metrics)
        record = {
            "epoch": epoch + 1,
            "train_loss": running / max(1, len(train_loader)),
            **values,
            **status,
        }
        history.append(record)
        print(
            f"\n[Epoch {epoch + 1}] Amp={values['amp_pred_ssim']:.4f} "
            f"(gain {values['amp_ssim_gain']:+.4f}) | "
            f"Phase-FG={values['phase_foreground_ssim']:.4f} | "
            f"GradCorr={values['phase_gradient_correlation']:.4f} | "
            f"BG-MAE={values['phase_background_mae_rad']:.4f} rad | "
            f"score={joint:.4f}"
        )

        if joint > best_joint:
            best_joint = joint
            best_metrics = metrics.copy()
            save_checkpoint(
                os.path.join(save_dir, "best_generator.pth"), model, optimizer,
                epoch + 1, metrics, required_ssim, required_gradient_corr,
                max_background_mae, img_size, num_stages, shared_prior,
                features,
                seed,
                class_balanced,
            )
            print(">>> 最佳联合 SSIM 模型已更新。")
        if save_every_epoch:
            save_checkpoint(
                os.path.join(
                    save_dir, "checkpoints", f"full_epoch_{epoch + 1}.pth"
                ),
                model, optimizer, epoch + 1, metrics, required_ssim,
                required_gradient_corr, max_background_mae, img_size,
                num_stages, shared_prior, features, seed, class_balanced,
            )
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, "checkpoints", f"epoch_{epoch + 1}.pth"))
        with open(os.path.join(save_dir, "training_history.json"), "w", encoding="utf-8") as stream:
            json.dump(history, stream, ensure_ascii=False, indent=2)

    if best_metrics is None:
        raise RuntimeError("训练没有产生有效验证结果。")
    final_status = acceptance_status(
        best_metrics, required_ssim, required_gradient_corr, max_background_mae
    )
    summary = {
        **metrics_dict(best_metrics),
        **final_status,
        "support_source": "amplitude_measurement_only",
        "uses_phase_mask": False,
        "phase_mask_used_for_evaluation_only": True,
        "phase_alignment": "none",
    }
    with open(os.path.join(save_dir, "validation_summary.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    status_text = "达到" if summary["threshold_met"] else "未达到"
    print(f"\n训练结束：严格验收={status_text}；selection score={best_joint:.4f}。")


def main() -> None:
    base_path = r"C:\Users\Lenovo\Desktop\Work\computing_imaging\GAN_SPS"
    parser = argparse.ArgumentParser(description="训练V11物理屏幕标定版可微NCPP。")
    parser.add_argument("--data-dir", default=os.path.join(base_path, "real_cell_dataset_v11_physical_screen"))
    parser.add_argument("--save-dir", default=os.path.join(base_path, "trained_models_real_cells_v11_physical_screen"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-stages", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--features", type=int, default=48)
    parser.add_argument(
        "--shared-prior", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--required-ssim", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--required-gradient-corr", type=float, default=0.75)
    parser.add_argument("--max-background-mae", type=float, default=0.05)
    parser.add_argument("--quiet", action="store_true", help="仅打印每轮验证摘要。")
    parser.add_argument(
        "--init-checkpoint", default=None,
        help="载入兼容权重后用新的优化器和损失继续训练。",
    )
    parser.add_argument(
        "--class-balanced", action=argparse.BooleanOptionalAction, default=True,
        help="按细胞类型逆频率采样，避免少数类被 PC3/PNT1A 淹没。",
    )
    parser.add_argument(
        "--a2058-boost", type=float, default=1.0,
        help="在类型均衡基础上额外提高 A2058 采样概率。",
    )
    parser.add_argument("--save-every-epoch", action="store_true")
    args = parser.parse_args()
    train_real_cells(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_stages=args.num_stages,
        num_workers=args.num_workers,
        shared_prior=args.shared_prior,
        required_ssim=args.required_ssim,
        features=args.features,
        seed=args.seed,
        required_gradient_corr=args.required_gradient_corr,
        max_background_mae=args.max_background_mae,
        quiet=args.quiet,
        init_checkpoint=args.init_checkpoint,
        class_balanced=args.class_balanced,
        a2058_boost=args.a2058_boost,
        save_every_epoch=args.save_every_epoch,
    )


if __name__ == "__main__":
    main()
