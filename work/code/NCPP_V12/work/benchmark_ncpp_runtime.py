"""Reproducible same-hardware latency benchmark for learned and classical NCPP."""

from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(r"C:\Users\Lenovo\Documents\Codex\2026-08-30\ncpp-ncpp-ncpp-annotated-quantitative-phase")
CODE = ROOT / "outputs" / "ncpp_physical_screen_v11"
DATA = ROOT / "work" / "real_cell_dataset_v11"
CHECKPOINT = ROOT / "outputs" / "ncpp_physical_screen_v11_results" / "best_generator.pth"
OUTPUT = ROOT / "work" / "paper_runtime_benchmark.json"
sys.path.insert(0, str(CODE))

from real_GAN_dataset import RelaySPSDataset  # noqa: E402
from real_model import PhysicsUnrolledGenerator, _camera_modulus_gain  # noqa: E402


def to_device_batch(sample, batch_size: int, device: torch.device):
    tensors = []
    for item in sample:
        tensor = item.unsqueeze(0).to(device)
        repeats = [batch_size] + [1] * (tensor.ndim - 1)
        tensors.append(tensor.repeat(*repeats))
    return tensors


def make_propagators(params_phys: torch.Tensor, n: int):
    lamb = params_phys[:, 0:1, None, None]
    z = params_phys[:, 1:2, None, None]
    l_s = params_phys[:, 2:3, None, None]
    dx_in = lamb * z / l_s
    dtype = params_phys.dtype
    coords = torch.arange(n, device=params_phys.device, dtype=dtype) - n // 2
    yy_base, xx_base = torch.meshgrid(coords, coords, indexing="ij")
    xx = xx_base[None, None] * dx_in
    yy = yy_base[None, None] * dx_in
    k = 2.0 * math.pi / lamb
    side_length = n * dx_in
    fresnel_number = (0.5 * side_length).square() / (lamb * z)
    fresnel_gate = (fresnel_number >= 0.01).to(dtype)
    phi_in_f = torch.polar(torch.ones_like(xx), k * (xx.square() + yy.square()) / (2.0 * z))
    phi_in = fresnel_gate * phi_in_f + (1.0 - fresnel_gate)

    dx_out = l_s / n
    xxo = xx_base[None, None] * dx_out
    yyo = yy_base[None, None] * dx_out
    phi_out_f = torch.polar(
        torch.ones_like(xxo), k * (xxo.square() + yyo.square()) / (2.0 * z)
    )
    phi_out = fresnel_gate * phi_out_f + (1.0 - fresnel_gate)
    scaling = dx_in.square() / (1j * lamb * z)

    def forward(field: torch.Tensor) -> torch.Tensor:
        return phi_out * torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(field * phi_in, dim=(-2, -1))),
            dim=(-2, -1),
        ) * scaling

    def inverse(field: torch.Tensor) -> torch.Tensor:
        spectrum = field * torch.conj(phi_out) / scaling
        return torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1))),
            dim=(-2, -1),
        ) * torch.conj(phi_in)

    return forward, inverse


@torch.inference_mode()
def classical_ncpp(
    sps: torch.Tensor,
    params_phys: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """Algorithm 2 schedule: 3 gamma × 11 beta × 31 inner cycles."""
    batch, _, n, _ = sps.shape
    forward, inverse = make_propagators(params_phys, n)
    s_min = params_phys[:, 3:4, None, None]
    s_max = params_phys[:, 4:5, None, None]
    camera_gain = _camera_modulus_gain(params_phys)
    hard_support = support >= 0.5
    generator = torch.Generator(device=sps.device)
    generator.manual_seed(20260901)
    field = torch.rand((batch, 1, n, n), generator=generator, device=sps.device)
    field = field.to(torch.complex64)
    field = torch.where(hard_support, field, torch.zeros_like(field))

    gammas = (1.2, 1.0, 0.8)
    betas = tuple(float(v) for v in np.linspace(3.0, 1.0, 11))
    for gamma in gammas:
        modulus = (sps.clamp(0.0, 1.0).pow(gamma) * (s_max - s_min) + s_min) / camera_gain
        for beta in betas:
            for _ in range(31):
                observed = forward(field)
                observed = modulus * observed / (observed.abs() + 1e-8)
                candidate = inverse(observed)
                # HIO support projection; complex values are retained inside the support.
                field = torch.where(hard_support, candidate, field - beta * candidate)

                observed = forward(field)
                observed = modulus * observed / (observed.abs() + 1e-8)
                candidate = inverse(observed)
                # ER support projection.
                field = torch.where(hard_support, candidate, torch.zeros_like(candidate))
    return field


def cuda_time_ms(fn, warmup: int, repeats: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop))
    values = np.asarray(samples, dtype=np.float64)
    return {
        "repeats": repeats,
        "mean_ms_per_batch": float(values.mean()),
        "sd_ms_per_batch": float(values.std(ddof=1)) if repeats > 1 else 0.0,
        "median_ms_per_batch": float(np.median(values)),
        "p05_ms_per_batch": float(np.quantile(values, 0.05)),
        "p95_ms_per_batch": float(np.quantile(values, 0.95)),
    }


def benchmark_batch(batch_size: int, dataset, model, device):
    sample = dataset[0]
    (
        sps,
        params_norm,
        params_phys,
        amp_meas,
        amp_flat,
        support,
        saturation,
        _target,
    ) = to_device_batch(sample, batch_size, device)

    model_fn = lambda: model(
        sps,
        params_norm,
        params_phys,
        amplitude_measurement=amp_meas,
        amplitude_flat_field=amp_flat,
        support_mask=support,
        saturation_mask=saturation,
        device=device,
    )
    classical_fn = lambda: classical_ncpp(sps, params_phys, support)
    learned = cuda_time_ms(model_fn, warmup=15, repeats=100)
    classical = cuda_time_ms(classical_fn, warmup=1, repeats=5)
    for item in (learned, classical):
        item["mean_ms_per_frame"] = item["mean_ms_per_batch"] / batch_size
        item["frames_per_second"] = 1000.0 / item["mean_ms_per_frame"]
    return {
        "batch_size": batch_size,
        "learned": learned,
        "classical_ncpp": classical,
        "wall_clock_speedup": classical["mean_ms_per_frame"] / learned["mean_ms_per_frame"],
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    torch.manual_seed(20260901)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    dataset = RelaySPSDataset(str(DATA), split="test")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = PhysicsUnrolledGenerator(
        num_stages=int(checkpoint["num_stages"]),
        features=int(checkpoint["features"]),
        img_size=int(checkpoint["img_size"]),
        shared_prior=bool(checkpoint["shared_prior"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    results = {
        "date": "2026-09-01",
        "device": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "pytorch": torch.__version__,
        "python": platform.python_version(),
        "grid": [dataset.image_size, dataset.image_size],
        "learned_stages": int(checkpoint["num_stages"]),
        "learned_parameter_count": sum(p.numel() for p in model.parameters()),
        "classical_schedule": {
            "gamma": [1.2, 1.0, 0.8],
            "beta": "3.0 to 1.0 inclusive, step -0.2 (11 values)",
            "inner_j": "0 to 30 inclusive (31 cycles)",
            "ping_pong_cycles": 3 * 11 * 31,
            "fft_or_ifft_count": 3 * 11 * 31 * 4,
        },
        "learned_fft_or_ifft_count": 1 + 2 * int(checkpoint["num_stages"]),
        "benchmark": [],
    }
    for batch_size in (1, 5):
        results["benchmark"].append(benchmark_batch(batch_size, dataset, model, device))
    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
