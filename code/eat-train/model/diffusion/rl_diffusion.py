from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import (
    cosine_beta_schedule,
    linear_beta_schedule,
    vp_beta_schedule,
    extract,
)

class RLDiffusionActor(nn.Module):
    """
    RL 用的 Diffusion 动作头：
    - cond_preprocess(obs) -> cond_state 特征
    - 反向扩散采样 T 步，UNet 返回噪声均值；实际噪声 = 均值 + sigma * N(0, I)
    - 输出最终 x_0 作为动作（mu），并返回一个与之等形状的 sigma 向量供策略构造分布与熵

    约定 UNet 模型签名：model(x_noisy, t, cond_state) -> noise_mean
    注意：本模块仅保留 RL 采样相关逻辑，不包含模仿/重建损失。
    """

    def __init__(
        self,
        cond_preprocess: nn.Module,
        action_dim: int,
        model: nn.Module,
        max_action: float = 1.0,
        n_timesteps: int = 10,
        beta_schedule: str = "linear",
        sigma_init: float = 0.1,
        action_low: Optional[torch.Tensor] = None,
        action_high: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.cond_preprocess = cond_preprocess
        self.action_dim = int(action_dim)
        self.output_dim = int(action_dim)  # 兼容 tianshou 的 preprocess_net 接口
        self.max_action = float(max_action)
        self.model = model

        # beta schedule
        if beta_schedule == "linear":
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == "vp":
            betas = vp_beta_schedule(n_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=alphas.dtype), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)

        # 注册 buffer
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # 可学习的标量：存储为 log_sigma，更稳定
        sigma_init = max(1e-6, float(sigma_init))
        self.log_sigma_param = nn.Parameter(torch.log(torch.tensor(sigma_init, dtype=torch.float32)))

        # 动作上下限：支持按维度设置；若未提供则按对称区间 [-max_action, max_action]
        if action_low is None or action_high is None:
            low = -float(max_action) * torch.ones(self.action_dim, dtype=torch.float32)
            high = float(max_action) * torch.ones(self.action_dim, dtype=torch.float32)
        else:
            low = action_low.float().view(-1)
            high = action_high.float().view(-1)
            assert low.numel() == self.action_dim and high.numel() == self.action_dim, "action_low/high shape mismatch"
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)

    # -------------- 内部函数 --------------
    def _sigma_value(self) -> torch.Tensor:
        # sigma = softplus(log_sigma) + eps，保证正数与稳定
        return F.softplus(self.log_sigma_param) + 1e-6

    def _predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def _q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _p_mean_variance(self, x: torch.Tensor, t: torch.Tensor, cond_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # UNet 预测噪声均值，然后加上 sigma * N(0,I) 得到真实噪声
        noise_mean = self.model(x, t, cond_state)
        noise = noise_mean # + self._sigma_value() * torch.randn_like(noise_mean)
        x_recon = self._predict_start_from_noise(x, t=t, noise=noise)
        # 将重建的 x0 映射到动作空间范围：先 tanh 到 [-1,1]，再按维缩放
        x_recon = self._squash_to_bounds(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self._q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def _p_sample(self, x: torch.Tensor, t: torch.Tensor, cond_state: torch.Tensor) -> torch.Tensor:
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self._p_mean_variance(x=x, t=t, cond_state=cond_state)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def _p_sample_loop(self, cond_state: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self._p_sample(x, timesteps, cond_state)
        # 最终输出再做一次边界映射，确保范围
        return self._squash_to_bounds(x)

    # -------------- 对外接口（tianshou actor 兼容） --------------
    def forward(
        self,
        obs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 获取条件特征
        cond_feat, _ = self.cond_preprocess(obs, state=None, info=info if info is not None else {})

        if not torch.is_tensor(cond_feat):
            cond_feat = torch.as_tensor(cond_feat, dtype=torch.float32, device=self.betas.device)

        batch_size = cond_feat.shape[0]
        action = self._p_sample_loop(cond_state=cond_feat, shape=(batch_size, self.action_dim))
        # 边界已在采样循环末尾处理，这里保守再确保一次
        action = self._squash_to_bounds(action)

        # 仅返回确定性动作（不返回分布参数）
        return action, state

    # 提供一个动作熵的便捷接口（标量/批次平均）
    def action_entropy(self, batch_size: int) -> torch.Tensor:
        # 多维高斯熵：H = dim * 0.5 * log(2*pi*e*sigma^2)
        sigma = self._sigma_value()
        dim = float(self.action_dim)
        ent = 0.5 * dim * torch.log(2.0 * torch.tensor(np.pi * np.e, device=sigma.device) * (sigma ** 2))
        # 返回 shape=[batch_size] 的张量，便于与 PPO 汇合
        return ent.expand(batch_size)

    # -------------- 辅助函数：动作范围映射 --------------
    def _squash_to_bounds(self, x: torch.Tensor) -> torch.Tensor:
        # x -> tanh(x) in [-1,1]
        x_unit = torch.tanh(x)
        # scale to [low, high] per-dimension
        # Broadcast low/high to batch shape
        low = self.action_low.view(1, -1).to(x_unit.device)
        high = self.action_high.view(1, -1).to(x_unit.device)
        mid = (low + high) * 0.5
        half = (high - low) * 0.5
        return mid + half * x_unit
