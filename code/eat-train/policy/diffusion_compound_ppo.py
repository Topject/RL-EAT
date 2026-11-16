from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from torch import nn
from torch.distributions import Categorical

from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy.modelfree.ppo import PPOPolicy


class CompositeCategorical:
    def __init__(self, exec_logits: torch.Tensor, step_logits: torch.Tensor, task_logits: torch.Tensor) -> None:
        self.exec_dist = Categorical(logits=exec_logits)
        self.step_dist = Categorical(logits=step_logits)
        self.task_dist = Categorical(logits=task_logits)

    def sample(self) -> torch.Tensor:
        a0 = self.exec_dist.sample()
        a1 = self.step_dist.sample()
        a2 = self.task_dist.sample()
        return torch.stack([a0, a1, a2], dim=-1)

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        a0, a1, a2 = act[..., 0], act[..., 1], act[..., 2]
        lp = self.exec_dist.log_prob(a0) + self.step_dist.log_prob(a1) + self.task_dist.log_prob(a2)
        return lp

    def entropy(self) -> torch.Tensor:
        return self.exec_dist.entropy() + self.step_dist.entropy() + self.task_dist.entropy()


class DiffusionCompoundPPOPolicy(PPOPolicy):
    """
    复合离散动作的 PPO：
    - actor 输出 concat logits，拆分为三头后构造 CompositeCategorical 分布
    - 其余 PPO 流程沿用父类
    """

    def _split_heads(self, logits_concat: torch.Tensor, exec_dim: int, step_dim: int, task_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e = exec_dim
        k = step_dim
        l = task_dim
        logits_exec = logits_concat[:, :e]
        logits_step = logits_concat[:, e:e+k]
        logits_task = logits_concat[:, e+k:e+k+l]
        return logits_exec, logits_step, logits_task

    def forward(self, batch: Batch, state: Optional[Dict[str, torch.Tensor]] = None, **kwargs: Any) -> Batch:  # type: ignore
        logits_concat, state = self.actor(batch.obs, state)  # actor: CompoundDiscreteDiffusion
        # 从 actor 或 action_space 获取三头维度，避免维度猜测
        e = getattr(self.actor, "exec_classes", None)
        k = getattr(self.actor, "step_classes", None)
        l = getattr(self.actor, "task_classes", None)
        if e is None or k is None or l is None:
            nvec = getattr(getattr(self, "action_space", None), "nvec", None)
            if nvec is None:
                raise RuntimeError("Cannot determine composite action head sizes: actor and action_space lack metadata.")
            e, k, l = int(nvec[0]), int(nvec[1]), int(nvec[2])
        logits_exec, logits_step, logits_task = self._split_heads(logits_concat, int(e), int(k), int(l))
        dist = CompositeCategorical(logits_exec, logits_step, logits_task)
        if self.training:
            act = dist.sample()
        else:
            a0 = torch.argmax(logits_exec, dim=-1)
            a1 = torch.argmax(logits_step, dim=-1)
            a2 = torch.argmax(logits_task, dim=-1)
            act = torch.stack([a0, a1, a2], dim=-1)
        return Batch(logits=logits_concat, act=act, state=state, dist=dist)

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: torch.Tensor) -> Batch:
        if self._recompute_adv:
            self._buffer, self._indices = buffer, indices
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        with torch.no_grad():
            batch.logp_old = self(batch).dist.log_prob(batch.act)
        return batch


