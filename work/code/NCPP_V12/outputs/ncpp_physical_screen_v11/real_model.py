"""物理展开式 rPMT 相位重建网络 V11（物理屏幕标定版）。

设计原则
--------
1. SPS 采样网格与重建网格保持一致，不以 2× 插值冒充真实高频。
2. 按论文 Algorithm 1/2 使用 M=(sqrt(I))^gamma，gamma 从 1.2 递减至 0.8；
   过曝光像素只提供模量下界。
3. 空间支持域仅来自独立振幅采集图。支持域外相位趋近 0，振幅回到照明
   背景，绝不把总复场错误清零。
4. Fresnel/Fraunhofer 选择由 Fresnel 数决定，不使用固定距离阈值。
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8
FRESNEL_NUMBER_THRESHOLD = 0.01


def _complex_exp(phase: torch.Tensor) -> torch.Tensor:
    return torch.polar(torch.ones_like(phase), phase)


def _camera_modulus_gain(params_phys: torch.Tensor) -> torch.Tensor:
    """每个样本从传播光场模量到相机电子计数模量的标定增益。"""
    if params_phys.ndim != 2 or params_phys.shape[1] < 7:
        raise ValueError(
            "V10 params_phys 必须包含 7 项：lambda、Z、Ls、Smin、Smax、"
            "Fresnel number、camera modulus gain。请重新生成数据。"
        )
    return torch.clamp(params_phys[:, 6:7], min=EPS).view(-1, 1, 1, 1)


def _coordinate_grid(
    n: int,
    m: int,
    dx: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成支持 batch-wise dx 的坐标网格。"""
    y = (torch.arange(n, device=device, dtype=dtype) - n // 2).view(1, 1, n, 1)
    x = (torch.arange(m, device=device, dtype=dtype) - m // 2).view(1, 1, 1, m)
    return y * dx, x * dx


def adaptive_forward_prop_vec(
    u_in: torch.Tensor,
    z: torch.Tensor,
    lamb: torch.Tensor,
    dx_in: torch.Tensor,
    device: torch.device | None = None,
    z_threshold: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """可微分单 FFT Fresnel/Fraunhofer 正向传播。"""
    if u_in.ndim != 4:
        raise ValueError(f"u_in 应为 B×C×H×W，实际 {tuple(u_in.shape)}")
    if not torch.is_complex(u_in):
        u_in = u_in.to(torch.complex64)

    device = u_in.device if device is None else device
    b, _, n, m = u_in.shape
    if n != m:
        raise ValueError("当前物理传播实现要求正方形采样。")

    real_dtype = u_in.real.dtype
    z_safe = torch.clamp(z.to(real_dtype), min=1e-6)
    lamb_safe = torch.clamp(lamb.to(real_dtype), min=1e-12)
    dx_safe = torch.clamp(dx_in.to(real_dtype), min=1e-12)
    k = 2.0 * math.pi / lamb_safe

    y, x = _coordinate_grid(n, m, dx_safe, device, real_dtype)
    side_length = n * dx_safe
    fresnel_num = (0.5 * side_length).square() / (lamb_safe * z_safe + EPS)
    fresnel_gate = (fresnel_num >= FRESNEL_NUMBER_THRESHOLD).to(real_dtype)
    phi_in_f = _complex_exp(k * (x * x + y * y) / (2.0 * z_safe))
    phi_in = fresnel_gate * phi_in_f + (1.0 - fresnel_gate)

    u_temp = u_in * phi_in
    u_freq = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(u_temp, dim=(-2, -1))),
        dim=(-2, -1),
    )

    dx_out = lamb_safe * z_safe / (n * dx_safe)
    y_out, x_out = _coordinate_grid(n, m, dx_out, device, real_dtype)
    phi_out_f = _complex_exp(k * (x_out * x_out + y_out * y_out) / (2.0 * z_safe))
    phi_out = fresnel_gate * phi_out_f + (1.0 - fresnel_gate)
    scaling = dx_safe.square() / (1j * lamb_safe * z_safe)
    return phi_out * u_freq * scaling, dx_out


def adaptive_inverse_prop_vec(
    u_out: torch.Tensor,
    z: torch.Tensor,
    lamb: torch.Tensor,
    dx_out: torch.Tensor,
    device: torch.device | None = None,
    z_threshold: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """与 adaptive_forward_prop_vec 配对的精确离散逆传播。"""
    if u_out.ndim != 4:
        raise ValueError(f"u_out 应为 B×C×H×W，实际 {tuple(u_out.shape)}")
    if not torch.is_complex(u_out):
        u_out = u_out.to(torch.complex64)

    device = u_out.device if device is None else device
    _, _, n, m = u_out.shape
    if n != m:
        raise ValueError("当前物理传播实现要求正方形采样。")

    real_dtype = u_out.real.dtype
    z_safe = torch.clamp(z.to(real_dtype), min=1e-6)
    lamb_safe = torch.clamp(lamb.to(real_dtype), min=1e-12)
    dx_out_safe = torch.clamp(dx_out.to(real_dtype), min=1e-12)
    dx_in = lamb_safe * z_safe / (n * dx_out_safe)
    k = 2.0 * math.pi / lamb_safe

    y, x = _coordinate_grid(n, m, dx_in, device, real_dtype)
    y_out, x_out = _coordinate_grid(n, m, dx_out_safe, device, real_dtype)
    side_length = n * dx_in
    fresnel_num = (0.5 * side_length).square() / (lamb_safe * z_safe + EPS)
    fresnel_gate = (fresnel_num >= FRESNEL_NUMBER_THRESHOLD).to(real_dtype)

    phi_in_f = _complex_exp(k * (x * x + y * y) / (2.0 * z_safe))
    phi_in = fresnel_gate * phi_in_f + (1.0 - fresnel_gate)
    phi_out_f = _complex_exp(k * (x_out * x_out + y_out * y_out) / (2.0 * z_safe))
    phi_out = fresnel_gate * phi_out_f + (1.0 - fresnel_gate)
    scaling = dx_in.square() / (1j * lamb_safe * z_safe)

    u_freq = u_out * torch.conj(phi_out) / scaling
    u_temp = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(u_freq, dim=(-2, -1))),
        dim=(-2, -1),
    )
    u_in = u_temp * torch.conj(phi_in)
    return u_in, dx_in


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvAct(channels, channels, 3, dilation=dilation)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(_group_count(channels), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.conv2(self.conv1(x)), inplace=True)


class FrequencyPreservingPrior(nn.Module):
    """
    双路径先验：U-Net 上下文路径 + 全分辨率细节路径。

    输入不仅包含 DC 复场与当前振幅/相位，还显式加入 3×3 和 7×7
    局部低通差分，避免所有细节都经过池化瓶颈。
    """

    def __init__(self, in_ch: int = 8, out_ch: int = 2, base_f: int = 32) -> None:
        super().__init__()
        enriched_ch = in_ch * 3
        self.stem = ConvAct(enriched_ch, base_f)

        self.detail = nn.Sequential(
            ResidualBlock(base_f, dilation=1),
            ResidualBlock(base_f, dilation=2),
            ResidualBlock(base_f, dilation=1),
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(base_f, base_f * 2, 3, stride=2, padding=1),
            nn.GroupNorm(_group_count(base_f * 2), base_f * 2),
            nn.SiLU(inplace=True),
            ResidualBlock(base_f * 2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_f * 2, base_f * 4, 3, stride=2, padding=1),
            nn.GroupNorm(_group_count(base_f * 4), base_f * 4),
            nn.SiLU(inplace=True),
            ResidualBlock(base_f * 4, dilation=2),
            ResidualBlock(base_f * 4, dilation=4),
        )

        self.up1 = ConvAct(base_f * 4 + base_f * 2, base_f * 2)
        self.up2 = ConvAct(base_f * 2 + base_f, base_f)
        self.fuse = nn.Sequential(
            ResidualBlock(base_f),
            nn.Conv2d(base_f, out_ch, 3, padding=1),
        )

    @staticmethod
    def _high_pass_features(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        low3 = F.avg_pool2d(x, 3, stride=1, padding=1)
        low7 = F.avg_pool2d(x, 7, stride=1, padding=3)
        return x - low3, low3 - low7

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        high1, high2 = self._high_pass_features(x)
        f0 = self.stem(torch.cat([x, high1, high2], dim=1))
        detail = self.detail(f0)

        f1 = self.down1(f0)
        f2 = self.down2(f1)
        u1 = F.interpolate(f2, size=f1.shape[-2:], mode="nearest")
        u1 = self.up1(torch.cat([u1, f1], dim=1))
        u2 = F.interpolate(u1, size=f0.shape[-2:], mode="nearest")
        u2 = self.up2(torch.cat([u2, f0], dim=1))
        return self.fuse(u2 + detail)


class InitNet(nn.Module):
    def __init__(self, in_ch: int = 4, features: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvAct(in_ch, features),
            ResidualBlock(features),
            nn.Conv2d(features, 2, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def radial_frequency_map(n: int, m: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype).view(1, 1, n, 1)
    x = torch.linspace(-1.0, 1.0, m, device=device, dtype=dtype).view(1, 1, 1, m)
    return torch.clamp(torch.sqrt(x.square() + y.square()) / math.sqrt(2.0), 0.0, 1.0)


def sps_reliability_map(
    sps_norm: torch.Tensor,
    stage_progress: float,
) -> torch.Tensor:
    """
    仅由实测 SPS 估计的软置信度。

    与 V9 的差异：不再给接近噪声底的像素设置非零径向下限，避免把
    背景扣除后被截为 0 的频点强制当成“真实零频谱”。对确有微弱响应的
    高频像素，则随展开阶段逐步提高权重。
    """
    b, _, n, m = sps_norm.shape
    flat = sps_norm.detach().flatten(2)
    noise_floor = torch.quantile(flat, 0.08, dim=2, keepdim=True).view(b, 1, 1, 1)
    q30 = torch.quantile(flat, 0.30, dim=2, keepdim=True).view(b, 1, 1, 1)
    noise_scale = torch.clamp(q30 - noise_floor, min=0.005)

    signal = torch.relu(sps_norm - noise_floor)
    snr_like = signal / (noise_scale + EPS)
    data_conf = snr_like / (1.0 + snr_like)

    radius = radial_frequency_map(n, m, sps_norm.device, sps_norm.dtype)
    p = float(max(0.0, min(1.0, stage_progress)))
    # 后期提高已检测到的外围弱信号权重，但不把未检测频点强制置零。
    release_gain = 1.0 + (0.25 + 1.25 * p) * radius.pow(1.5)
    confidence = torch.clamp(data_conf * release_gain, 0.0, 1.0)

    saturation = (sps_norm >= 0.995).to(sps_norm.dtype)
    confidence = confidence * (1.0 - 0.45 * saturation)
    return torch.clamp(confidence, 0.0, 1.0)


class AmplitudeDrivenSupport(nn.Module):
    """
    仅由当前预测振幅估计细胞软支持域和边界。

    设计要点
    --------
    1. 使用局部背景归一化，减弱高斯照明和缓慢亮度漂移的影响；
    2. 同时利用弱吸收残差和振幅梯度，兼容透明细胞低对比度情形；
    3. 输出连续软支持域，不做二值硬裁剪；
    4. 在物理展开中对输入振幅 ``detach`` 后使用，避免网络通过抹平振幅
       来投机性扩大或缩小支持域。
    """

    def __init__(
        self,
        background_kernel: int = 21,
        threshold_k: float = 1.25,
        temperature: float = 0.55,
    ) -> None:
        super().__init__()
        if background_kernel % 2 == 0:
            background_kernel += 1
        self.background_kernel = int(background_kernel)
        self.threshold_k = float(threshold_k)
        self.temperature = float(max(temperature, 1e-3))

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ) / 8.0
        sobel_y = sobel_x.t().contiguous()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _local_mean(self, x: torch.Tensor) -> torch.Tensor:
        p = self.background_kernel // 2
        x_pad = F.pad(x, (p, p, p, p), mode="reflect")
        return F.avg_pool2d(x_pad, self.background_kernel, stride=1)

    def _grad_mag(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.sobel_x.to(dtype=x.dtype), padding=1)
        gy = F.conv2d(x, self.sobel_y.to(dtype=x.dtype), padding=1)
        return torch.sqrt(gx.square() + gy.square() + EPS)

    @staticmethod
    def _robust_positive_z(x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        flat = x.flatten(2)
        median = torch.quantile(flat, 0.50, dim=2, keepdim=True).view(b, 1, 1, 1)
        abs_dev = torch.abs(flat - median.flatten(2))
        mad = torch.quantile(abs_dev, 0.50, dim=2, keepdim=True).view(b, 1, 1, 1)
        scale = (1.4826 * mad).clamp_min(1e-5)
        return torch.relu((x - median) / scale)

    def forward(self, amp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if amp.ndim != 4 or amp.shape[1] != 1:
            raise ValueError(f"amp 应为 B×1×H×W，实际 {tuple(amp.shape)}")

        background = self._local_mean(amp).clamp_min(1e-4)

        # 透明细胞通常表现为相对背景的微弱吸收；同时保留绝对残差，
        # 以兼容局部亮于背景的折射/干涉扰动。
        relative_absorption = torch.relu((background - amp) / background)
        relative_deviation = torch.abs(amp - background) / background
        residual_score = self._robust_positive_z(
            0.75 * relative_absorption + 0.25 * relative_deviation
        )

        normalized_amp = amp / background
        gradient_score = self._robust_positive_z(self._grad_mag(normalized_amp))
        object_score = 0.70 * residual_score + 0.30 * gradient_score

        soft_region = torch.sigmoid(
            (object_score - self.threshold_k) / self.temperature
        )
        # 先闭运算式连接弱轮廓，再平滑为连续支持概率。
        soft_region = F.max_pool2d(soft_region, 5, stride=1, padding=2)
        soft_region = F.avg_pool2d(soft_region, 5, stride=1, padding=2)
        soft_region = torch.clamp(soft_region, 0.0, 1.0)

        dilated = F.max_pool2d(soft_region, 3, stride=1, padding=1)
        eroded = -F.max_pool2d(-soft_region, 3, stride=1, padding=1)
        morph_boundary = torch.clamp(dilated - eroded, 0.0, 1.0)

        grad_boundary = torch.sigmoid(
            (gradient_score - self.threshold_k) / self.temperature
        )
        boundary = torch.clamp(
            0.55 * grad_boundary + 0.45 * morph_boundary,
            0.0,
            1.0,
        )
        return soft_region, boundary


class DataConsistencyBlock(nn.Module):
    def __init__(self, stage_index: int, num_stages: int, init_alpha: float = 0.8) -> None:
        super().__init__()
        self.stage_index = int(stage_index)
        self.num_stages = int(num_stages)
        init_alpha = min(max(init_alpha, 1e-3), 1.0 - 1e-3)
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(init_alpha / (1.0 - init_alpha))))

    @property
    def stage_progress(self) -> float:
        if self.num_stages <= 1:
            return 1.0
        return self.stage_index / float(self.num_stages - 1)

    def forward(
        self,
        amp_est: torch.Tensor,
        phase_est: torch.Tensor,
        sps_target: torch.Tensor,
        params_phys: torch.Tensor,
        amplitude_background: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
        saturation_mask: torch.Tensor | None = None,
        device: torch.device | None = None,
        z_threshold: float = 5.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = amp_est.device if device is None else device
        lamb = params_phys[:, 0:1].view(-1, 1, 1, 1)
        z = params_phys[:, 1:2].view(-1, 1, 1, 1)
        l_s = params_phys[:, 2:3].view(-1, 1, 1, 1)
        s_min = params_phys[:, 3:4].view(-1, 1, 1, 1)
        s_max = params_phys[:, 4:5].view(-1, 1, 1, 1)
        camera_gain = _camera_modulus_gain(params_phys)

        _, _, n, m = amp_est.shape
        if n != m or sps_target.shape[-2:] != (n, m):
            raise ValueError("DC 要求重建网格和 SPS 采样网格完全一致。")

        a_target = (sps_target * (s_max - s_min) + s_min) / camera_gain
        dx_in = lamb * z / l_s

        # adaptive_forward_prop_vec 已包含一次且仅一次 Fresnel 二次相位。
        # 旧实现此处再次乘 psi，导致训练前向模型与数据生成模型不一致。
        e_in = amp_est * _complex_exp(phase_est)
        e_obs, dx_out = adaptive_forward_prop_vec(
            e_in, z, lamb, dx_in, device=device, z_threshold=z_threshold
        )
        a_pred = torch.abs(e_obs)

        p = self.stage_progress
        # 论文 NCPP 的非线性 Fourier 模量：M=(sqrt(I))^gamma。
        # gamma>1 先压低弱高频、稳定轮廓；gamma<1 后释放弱高频细节。
        ncpp_gamma = 1.20 - 0.40 * p
        nonlinear_norm = torch.clamp(sps_target, 0.0, 1.0).pow(ncpp_gamma)
        a_candidate = (nonlinear_norm * (s_max - s_min) + s_min) / camera_gain

        saturation = (sps_target >= 0.995) if saturation_mask is None else (saturation_mask > 0.5)
        # 饱和值是下界：预测更大时不强行压回截断值。
        a_candidate = torch.where(saturation & (a_pred > a_target), a_pred, a_candidate)

        confidence = sps_reliability_map(sps_target, p)
        # 低 SNR 频点不能被当成精确零，但 NCPP 后期仍需利用它们。
        confidence_floor = 0.10 + 0.35 * p
        confidence = confidence_floor + (1.0 - confidence_floor) * confidence
        step = torch.sigmoid(self.alpha_logit) * confidence

        phase_factor = e_obs / (a_pred + EPS)
        e_updated = a_candidate * phase_factor
        e_final = e_obs + step * (e_updated - e_obs)

        e_obj, _ = adaptive_inverse_prop_vec(
            e_final, z, lamb, dx_out, device=device, z_threshold=z_threshold
        )
        amp_dc = torch.abs(e_obj)
        # 附件 NCPP 的非负对象约束；本数据集定义为正光程延迟且小于 pi，
        # 因而不把 [-pi,0) 的包裹角误当成合法目标。
        phase_dc = torch.clamp(torch.angle(e_obj), 0.0, math.pi)
        if support_mask is not None:
            support = torch.clamp(support_mask, 0.0, 1.0)
            # 振幅支持域是置信度而不是硬真值。弱吸收细胞可能在振幅中漏检，
            # 因而支持外仍保留至少 35% 的物理更新能力，让 SPS 数据一致性恢复
            # 这些区域；黑背景由监督损失和非负对象先验共同约束。
            projection_strength = 0.15 + 0.50 * p
            gate = 1.0 - projection_strength * (1.0 - support)
            if amplitude_background is None:
                amplitude_background = amp_est.detach()
            background = torch.clamp(amplitude_background, 0.0, 1.25)
            amp_dc = background + gate * (amp_dc - background)
            phase_dc = gate * phase_dc
        e_demod = amp_dc * _complex_exp(phase_dc)
        return e_demod.real, e_demod.imag


def amplitude_background_from_measurement(
    amplitude: torch.Tensor,
    kernel_size: int = 21,
) -> torch.Tensor:
    """可微分形态学闭运算，估计弱吸收目标下的照明背景。"""
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    padded = F.pad(amplitude, (pad, pad, pad, pad), mode="reflect")
    dilated = F.max_pool2d(padded, kernel_size, stride=1)
    padded_dilated = F.pad(dilated, (pad, pad, pad, pad), mode="reflect")
    closed = -F.max_pool2d(-padded_dilated, kernel_size, stride=1)
    return torch.clamp(closed, 0.0, 1.25)


class PhysicsUnrolledGenerator(nn.Module):
    def __init__(
        self,
        num_stages: int = 6,
        features: int = 48,
        img_size: int = 128,
        shared_prior: bool = False,
    ) -> None:
        super().__init__()
        if img_size % 4 != 0:
            raise ValueError("img_size 必须能被 4 整除。")
        self.num_stages = int(num_stages)
        self.high_res_size = int(img_size)  # 兼容旧训练脚本变量名；不再 2× 插值。
        self.shared_prior = bool(shared_prior)

        self.dc_blocks = nn.ModuleList(
            [DataConsistencyBlock(i, self.num_stages, init_alpha=0.80) for i in range(self.num_stages)]
        )
        self.init_net = InitNet(in_ch=7, features=features)

        y = torch.linspace(-1.0, 1.0, img_size)
        x = torch.linspace(-1.0, 1.0, img_size)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        aperture = torch.exp(-torch.pow(radius / 0.86, 12.0))
        broad_beam = (0.90 + 0.10 * torch.exp(-torch.pow(radius / 0.72, 2.0))) * aperture
        self.base_illumination = nn.Parameter(broad_beam.unsqueeze(0).unsqueeze(0))

        if self.shared_prior:
            self.prior = FrequencyPreservingPrior(in_ch=11, out_ch=2, base_f=32)
        else:
            self.prior_nets = nn.ModuleList(
                [FrequencyPreservingPrior(in_ch=11, out_ch=2, base_f=32) for _ in range(self.num_stages)]
            )
        self.update_scales = nn.Parameter(torch.full((self.num_stages, 2), -1.2))
        # 末端全分辨率细化器只修正最终相位，不参与中间 NCPP 数据一致性。
        # 最后一层零初始化，保证从旧物理展开权重迁移时初始输出完全不变。
        self.output_refiner = FrequencyPreservingPrior(
            in_ch=11, out_ch=1, base_f=features
        )
        nn.init.zeros_(self.output_refiner.fuse[-1].weight)
        nn.init.zeros_(self.output_refiner.fuse[-1].bias)
        self.refine_scale_logit = nn.Parameter(torch.tensor(-0.4))
        self.amplitude_support = AmplitudeDrivenSupport(
            background_kernel=21,
            threshold_k=1.25,
            temperature=0.55,
        )

    def _prior_at(self, index: int) -> nn.Module:
        return self.prior if self.shared_prior else self.prior_nets[index]

    def forward(
        self,
        sps_input: torch.Tensor,
        params_norm: torch.Tensor,
        params_phys: torch.Tensor,
        amplitude_measurement: torch.Tensor | None = None,
        amplitude_flat_field: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
        saturation_mask: torch.Tensor | None = None,
        device: torch.device | None = None,
        z_threshold: float = 5.0,
    ) -> List[torch.Tensor] | torch.Tensor:
        device = sps_input.device if device is None else device
        if sps_input.shape[-2:] != (self.high_res_size, self.high_res_size):
            raise ValueError(
                f"输入 SPS 尺寸必须为 {self.high_res_size}×{self.high_res_size}，"
                f"实际 {tuple(sps_input.shape[-2:])}。请重新生成数据，不要训练时插值。"
            )
        if params_norm.shape[1] < 3:
            raise ValueError("params_norm 至少应包含 lambda、Z_OS、log10(Ls) 三项。")
        if amplitude_measurement is not None and amplitude_measurement.shape != sps_input.shape:
            raise ValueError("amplitude_measurement 必须与 SPS 同尺寸。")
        if amplitude_flat_field is not None and amplitude_flat_field.shape != sps_input.shape:
            raise ValueError("amplitude_flat_field 必须与 SPS 同尺寸。")
        if support_mask is not None and support_mask.shape != sps_input.shape:
            raise ValueError("support_mask 必须与 SPS 同尺寸。")
        if saturation_mask is not None and saturation_mask.shape != sps_input.shape:
            raise ValueError("saturation_mask 必须与 SPS 同尺寸。")

        z = params_phys[:, 1:2].view(-1, 1, 1, 1)
        lamb = params_phys[:, 0:1].view(-1, 1, 1, 1)
        l_s = params_phys[:, 2:3].view(-1, 1, 1, 1)
        s_min = params_phys[:, 3:4].view(-1, 1, 1, 1)
        s_max = params_phys[:, 4:5].view(-1, 1, 1, 1)
        camera_gain = _camera_modulus_gain(params_phys)

        n = self.high_res_size
        dx_out = l_s / n
        a_sensor = (sps_input * (s_max - s_min) + s_min) / camera_gain

        e_init, _ = adaptive_inverse_prop_vec(
            a_sensor.to(torch.complex64), z, lamb, dx_out, device=device, z_threshold=z_threshold
        )
        sps_high = sps_input - F.avg_pool2d(sps_input, 5, stride=1, padding=2)
        sps_log = torch.log1p(100.0 * torch.clamp(sps_input, 0.0, 1.0)) / math.log(101.0)
        base = torch.clamp(self.base_illumination, 0.0, 1.10).expand(sps_input.shape[0], -1, -1, -1)
        has_amplitude_measurement = amplitude_measurement is not None
        if not has_amplitude_measurement:
            amp_measurement = base
            amplitude_background = base
        else:
            amp_measurement = torch.clamp(amplitude_measurement, 0.0, 1.20)
            amplitude_background = (
                amplitude_background_from_measurement(amp_measurement)
                if amplitude_flat_field is None
                else torch.clamp(amplitude_flat_field, 0.0, 1.25)
            )

        if support_mask is None and has_amplitude_measurement:
            support_from_amp, _ = self.amplitude_support(amp_measurement.detach())
        else:
            support_from_amp = support_mask
        support_for_features = (
            torch.full_like(sps_input, 0.5)
            if support_from_amp is None else torch.clamp(support_from_amp, 0.0, 1.0)
        )
        init_delta = self.init_net(
            torch.cat(
                [
                    e_init.real, e_init.imag, sps_input, sps_log, sps_high,
                    amp_measurement, support_for_features,
                ],
                dim=1,
            )
        )

        amp_est = torch.clamp(amp_measurement + 0.05 * init_delta[:, 0:1], 0.0, 1.20)
        phase_est = torch.clamp(0.50 * init_delta[:, 1:2], 0.0, math.pi)

        param_maps = params_norm[:, :3, None, None].expand(-1, -1, n, n)
        stage_preds: List[torch.Tensor] = []

        for i in range(self.num_stages):
            real_dc, imag_dc = self.dc_blocks[i](
                amp_est,
                phase_est,
                sps_input,
                params_phys,
                amplitude_background=amplitude_background,
                support_mask=support_from_amp,
                saturation_mask=saturation_mask,
                device=device,
                z_threshold=z_threshold,
            )
            progress = 1.0 if self.num_stages <= 1 else i / float(self.num_stages - 1)
            support_feature = 0.5 + (progress ** 2) * (support_for_features - 0.5)
            progress_map = torch.full_like(sps_input, progress)
            net_in = torch.cat(
                [
                    real_dc, imag_dc, amp_est, phase_est, param_maps,
                    support_feature, amp_measurement, sps_log, progress_map,
                ],
                dim=1,
            )
            delta = self._prior_at(i)(net_in)
            scales = 0.35 * torch.sigmoid(self.update_scales[i])

            # 最后一阶段也不把支持外更新关闭：support=0 时保留 35% 通道。
            support_strength = 0.65 * (progress ** 2)
            phase_update_gate = 1.0 - support_strength * (1.0 - support_for_features)

            amp_est = torch.clamp(
                amp_est + scales[0] * delta[:, 0:1],
                0.0,
                1.25,
            )
            phase_est = torch.clamp(
                phase_est + scales[1] * phase_update_gate * delta[:, 1:2],
                0.0,
                math.pi,
            )
            if support_from_amp is not None:
                gate = 1.0 - support_strength * (1.0 - support_for_features)
                amp_est = amplitude_background + gate * (amp_est - amplitude_background)
                phase_est = gate * phase_est
            if has_amplitude_measurement:
                # 振幅已是可采集量；以观测为锚，只学习小幅去噪/物理校正，
                # 避免相位先验网络反而破坏原本高 SSIM 的振幅测量。
                amp_est = 0.90 * amp_measurement + 0.10 * amp_est
            if self.training:
                stage_preds.append(torch.cat([amp_est, phase_est], dim=1))

        refine_input = torch.cat(
            [
                amp_est, phase_est, amp_measurement, support_for_features,
                sps_input, sps_log, e_init.real, e_init.imag, param_maps,
            ],
            dim=1,
        )
        phase_delta = self.output_refiner(refine_input)
        refine_scale = 0.35 * torch.sigmoid(self.refine_scale_logit)
        phase_est = torch.clamp(phase_est + refine_scale * phase_delta, 0.0, math.pi)
        if self.training:
            stage_preds[-1] = torch.cat([amp_est, phase_est], dim=1)

        result = torch.cat([amp_est, phase_est], dim=1)
        return stage_preds if self.training else result


class DeterministicPhysicalLoss(nn.Module):
    """仅由预测物场和实测 SPS 构造的频率均衡物理损失。"""

    def __init__(self, lambda_diff: float = 1.0) -> None:
        super().__init__()
        self.lambda_diff = float(lambda_diff)

    def forward(
        self,
        amp_pred: torch.Tensor,
        phase_pred: torch.Tensor,
        sps_target: torch.Tensor,
        params_phys: torch.Tensor,
        saturation_mask: torch.Tensor | None = None,
        device: torch.device | None = None,
        z_threshold: float = 5.0,
        stage_progress: float = 1.0,
    ) -> torch.Tensor:
        device = amp_pred.device if device is None else device
        lamb = params_phys[:, 0:1].view(-1, 1, 1, 1)
        z = params_phys[:, 1:2].view(-1, 1, 1, 1)
        l_s = params_phys[:, 2:3].view(-1, 1, 1, 1)
        s_min = params_phys[:, 3:4].view(-1, 1, 1, 1)
        s_max = params_phys[:, 4:5].view(-1, 1, 1, 1)
        camera_gain = _camera_modulus_gain(params_phys)

        _, _, n, m = amp_pred.shape
        if sps_target.shape[-2:] != (n, m):
            raise ValueError("物理损失要求 SPS 与预测位于同一采样网格。")
        dx_in = lamb * z / l_s

        e_obj = amp_pred * _complex_exp(phase_pred)
        e_obs, _ = adaptive_forward_prop_vec(
            e_obj, z, lamb, dx_in, device=device, z_threshold=z_threshold
        )
        a_pred = torch.abs(e_obs)
        a_target = (sps_target * (s_max - s_min) + s_min) / camera_gain

        scale = a_target.detach().mean(dim=(-2, -1), keepdim=True).clamp_min(EPS)
        log_pred = torch.log1p(a_pred / scale)
        log_target = torch.log1p(a_target / scale)

        radius = radial_frequency_map(n, m, amp_pred.device, amp_pred.dtype)
        reliability = sps_reliability_map(sps_target, stage_progress=stage_progress)

        # 对可靠频点拟合实测模量；对近噪声底频点仅施加“不可明显超过检测上界”
        # 的单边约束，避免将背景扣除后的 0 误当成真实零频谱。
        flat = sps_target.detach().flatten(2)
        noise_floor_n = torch.quantile(flat, 0.08, dim=2, keepdim=True).view(-1, 1, 1, 1)
        q30_n = torch.quantile(flat, 0.30, dim=2, keepdim=True).view(-1, 1, 1, 1)
        noise_sigma_n = torch.clamp(q30_n - noise_floor_n, min=0.005)
        upper_n = torch.clamp(noise_floor_n + 1.5 * noise_sigma_n, 0.0, 1.0)
        a_upper = (upper_n * (s_max - s_min) + s_min) / camera_gain
        log_upper = torch.log1p(a_upper / scale)

        p = float(max(0.0, min(1.0, stage_progress)))
        radial_gain = 1.0 + (0.50 + 1.00 * p) * radius
        saturation = (
            (sps_target >= 0.995).to(amp_pred.dtype)
            if saturation_mask is None else (saturation_mask > 0.5).to(amp_pred.dtype)
        )
        reliable_residual = (1.0 - saturation) * reliability * (log_pred - log_target)
        censored_residual = (1.0 - saturation) * (1.0 - reliability) * torch.relu(log_pred - log_upper)
        # 过曝光值只给出下界：预测低于截断模量才受罚。
        saturation_residual = saturation * torch.relu(log_target - log_pred)
        residual = radial_gain * (reliable_residual + censored_residual + saturation_residual)
        loss = torch.mean(torch.sqrt(residual.square() + 1e-6))
        return self.lambda_diff * loss


def _gaussian_kernel(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    g /= g.sum()
    return torch.outer(g, g)


def _ssim_map(img1: torch.Tensor, img2: torch.Tensor, data_range: float) -> torch.Tensor:
    window_size = 11
    channel = img1.shape[1]
    kernel2d = _gaussian_kernel(window_size, 1.5, img1.device, img1.dtype)
    kernel = kernel2d.view(1, 1, window_size, window_size).repeat(channel, 1, 1, 1)

    mu1 = F.conv2d(img1, kernel, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, kernel, padding=window_size // 2, groups=channel)
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=window_size // 2, groups=channel) - mu1.square()
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=window_size // 2, groups=channel) - mu2.square()
    sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size // 2, groups=channel) - mu1 * mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    return ((2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1.square() + mu2.square() + c1) * (sigma1_sq + sigma2_sq + c2) + EPS
    )


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float, float]:
    """兼容入口：按样本计算原始预测指标，不读取真值进行 piston 对齐。"""
    amp_p, phase_p = pred[:, 0:1], pred[:, 1:2]
    amp_t, phase_t = target[:, 0:1], target[:, 1:2]

    values = []
    for i in range(pred.shape[0]):
        ap, at = amp_p[i:i+1], amp_t[i:i+1]
        pp, pt = phase_p[i:i+1], phase_t[i:i+1]
        mse_a = F.mse_loss(ap, at).item()
        mse_p = F.mse_loss(pp, pt).item()
        phase_range = math.pi
        amp_range = 1.0
        psnr_a = 10.0 * math.log10((amp_range * amp_range) / (mse_a + 1e-10))
        psnr_p = 10.0 * math.log10((phase_range * phase_range) / (mse_p + 1e-10))
        ssim_a = _ssim_map(ap, at, data_range=amp_range).mean().item()
        ssim_p = _ssim_map(pp, pt, data_range=phase_range).mean().item()
        values.append((psnr_a, psnr_p, ssim_a, ssim_p))
    result = torch.tensor(values, dtype=torch.float64).mean(dim=0)
    return tuple(float(v.item()) for v in result)



class MeasurementDrivenBoundary(AmplitudeDrivenSupport):
    """
    向后兼容名称。V9.2 中边界严格由预测振幅提取，不再使用相位。

    ``forward`` 仍允许传入第二个 phase 参数，以兼容旧调用，但该参数会被忽略。
    """

    def __init__(
        self,
        background_kernel: int = 21,
        threshold_k: float = 1.25,
        temperature: float = 0.55,
    ) -> None:
        super().__init__(
            background_kernel=background_kernel,
            threshold_k=threshold_k,
            temperature=temperature,
        )

    def forward(
        self,
        amp: torch.Tensor,
        phase: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del phase
        return super().forward(amp)
