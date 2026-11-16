from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import torch
from torch import nn

from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy import BasePolicy
from torch.distributions import Normal
import numpy as np


class DiffusionPPOPolicy(BasePolicy):
    """
    PPOPolicy 的变体：
    - actor 为 RLDiffusionActor，forward 返回 (mu, sigma_vec)
    - 熵项不使用 dist.entropy()，而是来自 actor.action_entropy(batch_size)
    - 其余流程与 PPOPolicy 一致，保留 ratio/clip 等
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        deterministic_eval: bool = True,
        # BasePolicy expected args:
        observation_space: Optional[Any] = None,
        action_space: Optional[Any] = None,
        action_scaling: bool = False,
        action_bound_method: str = "",
        lr_scheduler: Optional[Any] = None,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,
        )
        self.actor = actor
        self.critic = critic
        self.optim = optim
        self._gamma = discount_factor
        self._gae_lambda = gae_lambda
        self._eps_clip = eps_clip
        self._dual_clip = None
        self._value_clip = value_clip
        self._norm_adv = advantage_normalization
        self._recompute_adv = recompute_advantage
        self._weight_vf = vf_coef
        self._weight_ent = ent_coef
        self._grad_norm = max_grad_norm
        self._eps = 1e-8
        self._deterministic_eval = deterministic_eval
        self._action_scaling = action_scaling

    def set_timeout(self, **kwargs: Any) -> None:
        return

    # 计算 GAE / returns（从 PPOPolicy 迁移的等价实现）
    def _compute_returns(self, batch: Batch, buffer: ReplayBuffer, indices: torch.Tensor) -> Batch:
        v_ = self.critic(batch.obs).flatten()
        with torch.no_grad():
            next_v = self.critic(batch.obs_next).flatten()
        rew = to_torch_as(batch.rew, v_)
        term = to_torch_as(batch.terminated.astype(np.float32) if hasattr(batch, 'terminated') else batch.done.astype(np.float32), v_)
        batch.v_s = v_
        batch.returns = rew + self._gamma * (1.0 - term) * next_v
        batch.adv = batch.returns - v_
        return batch

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: torch.Tensor) -> Batch:
        if self._recompute_adv:
            self._buffer, self._indices = buffer, indices
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        with torch.no_grad():
            action_old = self(batch).act  # 确定性动作
            # 代理分布：仅用于 log_prob/ratio，不用于采样
            sigma = torch.tensor(0.1, device=action_old.device, dtype=action_old.dtype)
            dist = Normal(loc=action_old, scale=sigma)
            batch.logp_old = dist.log_prob(batch.act).sum(dim=-1)
        return batch

    def learn(self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any) -> Dict[str, List[float]]:
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        for step in range(repeat):
            if self._recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)
            for minibatch in batch.split(batch_size, merge_last=True):
                # 前向得到当前确定性动作
                action_now = self(minibatch).act
                # 代理分布，仅用于 log_prob/ratio
                sigma = torch.tensor(0.1, device=action_now.device, dtype=action_now.dtype)
                dist = Normal(loc=action_now, scale=sigma)
                if self._norm_adv:
                    mean, std = minibatch.adv.mean(), minibatch.adv.std()
                    minibatch.adv = (minibatch.adv - mean) / (std + self._eps)
                ratio = (dist.log_prob(minibatch.act).sum(dim=-1) - minibatch.logp_old).exp().float()
                surr1 = ratio * minibatch.adv
                surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip) * minibatch.adv
                if self._dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self._dual_clip * minibatch.adv)
                    clip_loss = -torch.where(minibatch.adv < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()

                # critic
                value = self.critic(minibatch.obs).flatten()
                if self._value_clip:
                    v_clip = minibatch.v_s + (value - minibatch.v_s).clamp(-self._eps_clip, self._eps_clip)
                    vf1 = (minibatch.returns - value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (minibatch.returns - value).pow(2).mean()

                # 熵：常数 sigma 的高斯熵（仅用于正则，不参与采样）
                with torch.no_grad():
                    dim = float(minibatch.act.shape[-1])
                    ent = 0.5 * dim * torch.log(2.0 * torch.tensor(3.1415926535 * 2.7182818284, device=action_now.device) * (sigma ** 2))

                loss = clip_loss + self._weight_vf * vf_loss - self._weight_ent * ent
                self.optim.zero_grad()
                loss.backward()
                if self._grad_norm:
                    nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), max_norm=self._grad_norm)
                self.optim.step()

                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent.item())
                losses.append(loss.item())

        return {
            "loss": losses,
            "loss/clip": clip_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }

    def forward(self, batch: Batch, state: Optional[Dict[str, torch.Tensor]] = None, **kwargs: Any) -> Batch:  # type: ignore
        action, _ = self.actor(batch.obs, state=state, info=getattr(batch, "info", None))
        return Batch(act=action, state=state, logits=action)


