from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, List

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
    基于 DDIM 的强化学习动作头：
    - cond_preprocess(obs) -> cond_state 特征
    - 反向扩散采样 T 步，UNet 返回噪声均值
    - 通过可学习的 eta 参数控制采样过程的随机性，eta 作为策略熵的代理
    - 支持在推理（forward）时传入更少的步数以加速采样

    约定 UNet 模型签名：model(x_noisy, t, cond_state) -> noise_mean
    """

    def __init__(
        self,
        cond_preprocess: nn.Module,
        action_dim: int,
        model: nn.Module,
        max_action: float = 1.0,
        n_timesteps: int = 10,
        beta_schedule: str = "linear",
        # eta_init: float = 1.0, # eta_param 初始化，eta 通过 sigmoid 映射到 [0, 1]
        action_low: Optional[torch.Tensor] = None,
        action_high: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.cond_preprocess = cond_preprocess
        self.action_dim = int(action_dim)
        self.output_dim = int(action_dim)
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
        # 在 alphas_cumprod 前面加上 1.0，方便处理 t=0 的情况
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=alphas.dtype), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)

        # 注册 buffer
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # --- 新增：可学习的 eta 参数，用于控制 DDIM 采样的随机性 ---
        # 原始的 eta_param 值可以设为 0，这样 sigmoid(0) = 0.5，eta 从中间值开始学习
        self.eta_param = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        # 动作上下限
        if action_low is None or action_high is None:
            low = -float(max_action) * torch.ones(self.action_dim, dtype=torch.float32)
            high = float(max_action) * torch.ones(self.action_dim, dtype=torch.float32)
        else:
            low = action_low.float().view(-1)
            high = action_high.float().view(-1)
            assert low.numel() == self.action_dim and high.numel() == self.action_dim, "action_low/high shape mismatch"
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)
        
        # --- 新增：用于统计噪声的字典 ---
        # 存储每次 forward 调用生成的初始噪声
        self.collected_initial_noises: List[torch.Tensor] = []
        # 存储每次 forward 调用中，每个 DDIM 步骤的 eta_controlled 噪声
        self.collected_eta_noises: List[List[torch.Tensor]] = []

    # -------------- 内部函数 --------------
    def _predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def _ddim_sample(
        self, x: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, cond_state: torch.Tensor,
        current_eta_noise_list: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        # 获取 eta 值，通过 sigmoid 约束在 [0, 1]
        eta = torch.sigmoid(self.eta_param)

        # 1. 预测噪声和 x0
        pred_noise = self.model(x, t, cond_state)
        x0_pred = self._predict_start_from_noise(x, t=t, noise=pred_noise)
        x0_pred.clamp_(-self.max_action, self.max_action)

        # 2. 获取所需的 alphas，这次我们显式地使用 t 和 t_prev
        # alpha_t_prev 不再从那个捷径 buffer 中获取
        alpha_t = extract(self.alphas_cumprod, t, x.shape).squeeze(-1)
        
        # +++ 核心改变在这里 +++
        # 创建一个与 t 形状相同，但值为 1.0 的张量，用于 t_prev = -1 的情况
        alpha_t_prev = torch.ones_like(t, dtype=torch.float32)
        # 找到 t_prev >= 0 的有效索引
        valid_indices = (t_prev >= 0)
        # 仅对有效索引，从 alphas_cumprod 中提取相应的值
        alpha_t_prev[valid_indices] = extract(self.alphas_cumprod, t_prev[valid_indices], x.shape).squeeze(-1)

        # 3. 计算 sigma (随机性的大小)
        # 这个公式现在可以处理任意的 t 和 t_prev 了
        variance = (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
        sigma = eta * torch.sqrt(variance)
        
        # 4. 计算指向 x_t 的方向的导数 (pred_epsilon)
        direction_to_x = torch.sqrt(1 - alpha_t_prev - sigma**2 + 1e-6).unsqueeze(1) * pred_noise
        
        # 检查sqrt的输入是否存在0
        # if (1 - alpha_t_prev - sigma**2 <= 0).any():
        #     raise ValueError("1 - alpha_t_prev - sigma**2 < 0")
       
        # if (1 - alpha_t_prev - sigma**2 <= 1e-6).any():
        #     raise ValueError("1 - alpha_t_prev - sigma**2 < eps")
        
        # 5. 计算 x_{t-1}
        # x0_pred [b,3]
        # direction_to_x [b,3]
        # 将alpha_t_prev从[8]扩展到[8,3]，以便进行后续计算
        alpha_t_prev = alpha_t_prev.unsqueeze(-1)
        
        x_prev = torch.sqrt(alpha_t_prev) * x0_pred + direction_to_x
        
        # 只有在 eta > 0 且 t > 0 时才添加随机噪声
        if eta > 0:
            noise = torch.randn_like(x)
            # t=0 时不加随机噪声
            nonzero_mask = (1 - (t == 0).float()).reshape(x.shape[0], *((1,) * (len(x.shape) - 1)))
            sigma = sigma.unsqueeze(-1)
            x_prev += sigma * nonzero_mask * noise

            eta_controlled_noise = sigma * nonzero_mask * noise
            x_prev += eta_controlled_noise

            # --- 收集 eta 控制的噪声 ---
            if current_eta_noise_list is not None:
                current_eta_noise_list.append(eta_controlled_noise.detach().cpu())

        if torch.isnan(x_prev).any():
            raise ValueError("x_prev is nan")

        return x_prev

    def _p_sample_loop(
        self, cond_state: torch.Tensor, shape: Sequence[int], n_inference_steps: int
    ) -> torch.Tensor:
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        # --- 收集初始噪声 ---
        self.collected_initial_noises.append(x.detach().cpu())
        
        # 创建一个临时列表来存储当前 forward 调用的 eta 噪声
        current_run_eta_noises: List[torch.Tensor] = []

        # 生成采样时间步序列
        times = torch.linspace(
            self.n_timesteps - 1, 0, steps=n_inference_steps, device=device, dtype=torch.long
        )
        times_prev = torch.cat([times[1:], torch.tensor([-1], device=device, dtype=torch.long)])

        for t, t_prev in zip(times, times_prev):
            # 扩展 t 和 t_prev 到 batch_size
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            t_prev_batch = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
            
            x = self._ddim_sample(x, t_batch, t_prev_batch, cond_state, 
                                  current_eta_noise_list=current_run_eta_noises)

        # --- 将当前 forward 调用的 eta 噪声列表添加到全局列表 ---
        self.collected_eta_noises.append(current_run_eta_noises)

        return self._squash_to_bounds(x)

    # -------------- 对外接口 --------------
    def forward(
        self,
        obs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        info = info if info is not None else {}
        # 从 info 中获取推理步数，若无则使用最大步数（训练模式）
        n_steps = info.get("n_inference_steps", self.n_timesteps)

        cond_feat, _ = self.cond_preprocess(obs, state=None, info=info)
        if not torch.is_tensor(cond_feat):
            cond_feat = torch.as_tensor(cond_feat, dtype=torch.float32, device=self.betas.device)

        batch_size = cond_feat.shape[0]
        action = self._p_sample_loop(
            cond_state=cond_feat, shape=(batch_size, self.action_dim), n_inference_steps=n_steps
        )
        
        # 边界已在采样循环末尾处理，这里保守再确保一次
        action = self._squash_to_bounds(action)

        # 返回确定性动作
        return action, state

    def action_entropy(self, batch_size: int) -> torch.Tensor:
        """
        将 eta 作为策略熵的代理。
        SAC 等算法会最大化此项，从而鼓励 eta 变大，增强探索。
        """
        # eta 通过 sigmoid 保证在 [0, 1]
        eta = torch.sigmoid(self.eta_param)
        
        # 直接使用 eta 作为熵代理。eta 越大，随机性越强，熵代理也越大。
        entropy_proxy = eta
        
        return entropy_proxy.expand(batch_size)

    # -------------- 辅助函数：动作范围映射 --------------
    def _squash_to_bounds(self, x: torch.Tensor) -> torch.Tensor:
        x_unit = torch.tanh(x)
        low = self.action_low.view(1, -1).to(x_unit.device)
        high = self.action_high.view(1, -1).to(x_unit.device)
        mid = (low + high) * 0.5
        half = (high - low) * 0.5
        return mid + half * x_unit

    def pop_noise_stats(self) -> Dict[str, Any]:
        """
        返回所有已收集的噪声数据，并清空内部存储。
        
        Returns:
            Dict[str, Any]: 包含以下键：
                - 'initial_noises': List[torch.Tensor]，每次 forward 调用开始时生成的初始噪声。
                - 'eta_noises': List[List[torch.Tensor]]，每次 forward 调用中，每个 DDIM 步骤的 eta_controlled 噪声。
        """
        stats = {
            "initial_noises": self.collected_initial_noises,
            "eta_noises": self.collected_eta_noises,
        }
        # 清空已储存的数据
        self.collected_initial_noises = []
        self.collected_eta_noises = []
        return stats
