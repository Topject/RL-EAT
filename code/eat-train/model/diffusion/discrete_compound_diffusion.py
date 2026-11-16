from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .helpers import (
    cosine_beta_schedule,
    linear_beta_schedule,
    vp_beta_schedule,
    extract,
)


class DiscreteDiffusionMLP(nn.Module):
    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        time_embed_dim: int = 32,
        hidden_sizes: Sequence[int] = (256, 256),
        device: str = "cpu",
    ) -> None:
        super().__init__()
        from tianshou.utils.net.common import MLP  # lazy import to avoid cycles
        self.time_embed_dim = int(time_embed_dim)
        self.t_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.ReLU(inplace=True),
        )
        # 输入: x_noisy (output_dim) + cond_state (state_dim) + t_embed (time_embed_dim)
        input_dim = int(output_dim + state_dim + self.time_embed_dim)
        self.mlp = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            device=device,
        )

    def build_time_embed(self, t: torch.Tensor) -> torch.Tensor:
        # sinusoidal embedding (manual implementation to avoid extra deps)
        device = t.device
        half = self.time_embed_dim // 2
        freq = torch.exp(
            torch.arange(half, device=device).float()
            * -(torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        args = t.float().unsqueeze(-1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.time_embed_dim:
            # pad if odd
            emb = torch.nn.functional.pad(emb, (0, self.time_embed_dim - emb.shape[-1]))
        return self.t_embed(emb)

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, cond_state: torch.Tensor) -> torch.Tensor:
        t_embed = self.build_time_embed(t)
        combined = torch.cat([x_noisy, cond_state, t_embed], dim=1)
        return self.mlp(combined)


class CompoundDiscreteDiffusion(nn.Module):
    """
    复合离散动作的 Diffusion 头：输出 concat logits，长度为 (2 + K + L)。
    - cond_preprocess: 预处理网络，forward(obs) -> (feat, state)
    - backbone: 噪声均值预测网络 DiscreteDiffusionMLP
    """

    def __init__(
        self,
        cond_preprocess: nn.Module,
        exec_classes: int,
        step_classes: int,
        task_classes: int,
        backbone: nn.Module,
        n_timesteps: int = 5,
        beta_schedule: str = "linear",
        logits_clip: float = 10.0,
    ) -> None:
        super().__init__()
        self.cond_preprocess = cond_preprocess
        self.exec_classes = int(exec_classes)
        self.step_classes = int(step_classes)
        self.task_classes = int(task_classes)
        self.total_dim = self.exec_classes + self.step_classes + self.task_classes
        self.output_dim = self.total_dim  # 兼容 tianshou preprocess 接口
        self.backbone = backbone
        self.logits_clip = float(logits_clip)

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

    # diffusion helpers
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
        noise_mean = self.backbone(x, t, cond_state)
        noise = noise_mean  # 不引入额外 sigma，直接用均值作为预测噪声
        x_recon = self._predict_start_from_noise(x, t=t, noise=noise)
        if self.logits_clip > 0:
            x_recon = torch.clamp(x_recon, -self.logits_clip, self.logits_clip)
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
        return x

    # tianshou preprocess 风格接口
    def forward(
        self,
        obs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cond_feat, _ = self.cond_preprocess(obs, state=None, info=info or {})
        if not torch.is_tensor(cond_feat):
            cond_feat = torch.as_tensor(cond_feat, dtype=torch.float32, device=self.betas.device)
        batch_size = cond_feat.shape[0]
        logits = self._p_sample_loop(cond_state=cond_feat, shape=(batch_size, self.total_dim))
        return logits, state


