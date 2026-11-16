from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np
import torch
import torch.nn as nn

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.policy import BasePolicy
from utils import calculate_gradient_metrics

# 无熵项
class SACDDPMPolicy(BasePolicy):
    """
    Soft Actor-Critic variant for diffusion-based actors producing deterministic actions.

    Differences from standard SAC:
    - Actor outputs concrete actions sampled inside a diffusion process (multi-modal), not a parametric policy.
    - Entropy term uses a global, learnable sigma from the diffusion actor: H = dim * 0.5 * log(2*pi*e*sigma^2).
      We substitute E[log pi(a|s)] with -H in both target and actor losses.

    Expected actor interface: actor(obs, state=None, info=None) -> (action, state)
    Additionally, actor.action_entropy(batch_size) -> Tensor[batch_size]
    """

    def __init__(
        self,
        actor: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1: nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2: nn.Module,
        critic2_optim: torch.optim.Optimizer,
        action_space=None,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        estimation_step: int = 1,
        lr_scheduler: Optional[Union[torch.optim.lr_scheduler.LambdaLR, Any]] = None,
        grad_clip_actor_max_norm: Optional[float] = None,
        grad_clip_critic_max_norm: Optional[float] = None,
    ) -> None:
        super().__init__(
            observation_space=None,
            action_space=action_space,
            action_scaling=False,
            action_bound_method="",
            lr_scheduler=lr_scheduler,
        )
        self.actor = actor
        self.actor_optim = actor_optim
        self.critic1 = critic1
        self.critic1_old = deepcopy(critic1).eval()
        self.critic1_optim = critic1_optim
        self.critic2 = critic2
        self.critic2_old = deepcopy(critic2).eval()
        self.critic2_optim = critic2_optim

        self.tau = float(tau)
        self.gamma = float(gamma)
        self.estimation_step = int(estimation_step)
        self.grad_clip_actor_max_norm = float(grad_clip_actor_max_norm) if grad_clip_actor_max_norm is not None else None
        self.grad_clip_critic_max_norm = float(grad_clip_critic_max_norm) if grad_clip_critic_max_norm is not None else None

        self._is_auto_alpha = False
        self._alpha: Union[float, torch.Tensor]
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            assert self._log_alpha.shape == torch.Size([1]) and self._log_alpha.requires_grad
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = float(alpha)

    def train(self, mode: bool = True) -> "SACDDPMPolicy":
        self.training = mode
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    @torch.no_grad()
    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]
        obs_next = batch.obs_next
        # Generate next action using diffusion actor
        info_next = getattr(batch, "info", None)
        act_next, _ = self.actor(obs_next, state=None, info=info_next)
        # Min double Q
        target_q = torch.min(
            self.critic1_old(obs_next, act_next),
            self.critic2_old(obs_next, act_next),
        )
        # Entropy bonus from global sigma
        batch_size = act_next.shape[0]
        entropy = self.actor.action_entropy(batch_size)  # shape [B]
        # In SAC target: r + gamma * ( Q_min(s', a') - alpha * E[log pi] )
        # Substitute E[log pi] with -H  =>  target += alpha * H
        # target_q = target_q + self._alpha * entropy.view(-1, 1)
        return target_q

    def sync_weight(self) -> None:
        self.soft_update(self.critic1_old, self.critic1, self.tau)
        self.soft_update(self.critic2_old, self.critic2, self.tau)

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        obs = batch[input]
        info_cur = getattr(batch, "info", None)
        act, hidden = self.actor(obs, state=state, info=info_cur)
        return Batch(logits=None, act=act, state=hidden, policy=Batch())

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        # Compute 1-step TD target with custom target Q definition
        batch = self.compute_nstep_return(
            batch,
            buffer,
            indices,
            target_q_fn=self._target_q,
            gamma=self.gamma,
            n_step=self.estimation_step,
            rew_norm=False,
        )
        return batch

    def _mse_optimizer(self, batch: Batch, critic: nn.Module, critic_optim: torch.optim.Optimizer) -> Tuple[torch.Tensor, torch.Tensor]:
        current_q = critic(batch.obs, batch.act).flatten()
        target_q = batch.returns.flatten()
        td = target_q - current_q
        critic_loss = (td.pow(2) * getattr(batch, "weight", 1.0)).mean()
        critic_optim.zero_grad()
        critic_loss.backward()
        # Gradient clipping to avoid exploding gradients in critic
        if self.grad_clip_critic_max_norm is not None:
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=self.grad_clip_critic_max_norm)
        critic_optim.step()
        return td.detach(), critic_loss

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # Update critics
        td1, critic1_loss = self._mse_optimizer(batch, self.critic1, self.critic1_optim)
        td2, critic2_loss = self._mse_optimizer(batch, self.critic2, self.critic2_optim)
        batch.weight = (td1 + td2) / 2.0  # for prioritized replay

        # Update actor: minimize -Q_min(s, a) - alpha * H(sigma)
        obs = batch.obs
        info_cur = getattr(batch, "info", None)
        act, _ = self.actor(obs, state=None, info=info_cur)
        q1 = self.critic1(obs, act).flatten()
        q2 = self.critic2(obs, act).flatten()
        q_min = torch.min(q1, q2)
        entropy = self.actor.action_entropy(batch_size=act.shape[0]).flatten()
        entropy = 0
        actor_loss = (-q_min - self._alpha * entropy).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Optional: automatic entropy tuning using -H as surrogate of E[log pi]
        if hasattr(self, "_is_auto_alpha") and self._is_auto_alpha:
            log_prob_surrogate = (-entropy).detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_prob_surrogate).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()
        else:
            alpha_loss = None

        # Soft-update target critics
        self.sync_weight()

        result: Dict[str, float] = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }
        if alpha_loss is not None:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = to_numpy(self._alpha)[()] if isinstance(self._alpha, torch.Tensor) else float(self._alpha)
        return result

class SACDDIMPolicy(BasePolicy):
    """
    Soft Actor-Critic variant for diffusion-based actors producing deterministic actions.

    Differences from standard SAC:
    - Actor outputs concrete actions sampled inside a diffusion process (multi-modal), not a parametric policy.
    - Entropy term uses a global, learnable sigma from the diffusion actor: H = dim * 0.5 * log(2*pi*e*sigma^2).
      We substitute E[log pi(a|s)] with -H in both target and actor losses.

    Expected actor interface: actor(obs, state=None, info=None) -> (action, state)
    Additionally, actor.action_entropy(batch_size) -> Tensor[batch_size]
    """

    def __init__(
        self,
        actor: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1: nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2: nn.Module,
        critic2_optim: torch.optim.Optimizer,
        action_space=None,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        estimation_step: int = 1,
        lr_scheduler: Optional[Union[torch.optim.lr_scheduler.LambdaLR, Any]] = None,
        grad_clip_actor_max_norm: Optional[float] = None,
        grad_clip_critic_max_norm: Optional[float] = None,
    ) -> None:
        super().__init__(
            observation_space=None,
            action_space=action_space,
            action_scaling=False,
            action_bound_method="",
            lr_scheduler=lr_scheduler,
        )
        self.actor = actor
        self.actor_optim = actor_optim
        self.critic1 = critic1
        self.critic1_old = deepcopy(critic1).eval()
        self.critic1_optim = critic1_optim
        self.critic2 = critic2
        self.critic2_old = deepcopy(critic2).eval()
        self.critic2_optim = critic2_optim

        self.tau = float(tau)
        self.gamma = float(gamma)
        self.estimation_step = int(estimation_step)

        self._is_auto_alpha = False
        self._alpha: Union[float, torch.Tensor]
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            assert self._log_alpha.shape == torch.Size([1]) and self._log_alpha.requires_grad
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = float(alpha)

        # Accumulators for monitoring q_min during training (per-epoch)
        self._min_q_sum_epoch: float = 0.0
        self._min_q_count_epoch: int = 0
        
        self._learn_step_counter: int = 0
        self._gradient_metrics_cache: List[Dict[str, Any]] = []

        self.grad_clip_actor_max_norm = float(grad_clip_actor_max_norm) if grad_clip_actor_max_norm is not None else None
        self.grad_clip_critic_max_norm = float(grad_clip_critic_max_norm) if grad_clip_critic_max_norm is not None else None


    def train(self, mode: bool = True) -> "SACDDPMPolicy":
        self.training = mode
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    @torch.no_grad()
    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]
        obs_next = batch.obs_next
        # Generate next action using diffusion actor
        info_next = getattr(batch, "info", None)
        act_next, _ = self.actor(obs_next, state=None, info=info_next)
        # Min double Q
        target_q = torch.min(
            self.critic1_old(obs_next, act_next),
            self.critic2_old(obs_next, act_next),
        )
        # Entropy bonus from global sigma
        batch_size = act_next.shape[0]
        entropy = self.actor.action_entropy(batch_size)  # shape [B]
        # In SAC target: r + gamma * ( Q_min(s', a') - alpha * E[log pi] )
        # Substitute E[log pi] with -H  =>  target += alpha * H
        # target_q = target_q + self._alpha * entropy.view(-1, 1)
        return target_q

    def sync_weight(self) -> None:
        self.soft_update(self.critic1_old, self.critic1, self.tau)
        self.soft_update(self.critic2_old, self.critic2, self.tau)

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        obs = batch[input]
        info_cur = getattr(batch, "info", None)
        act, hidden = self.actor(obs, state=state, info=info_cur)
        return Batch(logits=None, act=act, state=hidden, policy=Batch())

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        # Compute 1-step TD target with custom target Q definition
        batch = self.compute_nstep_return(
            batch,
            buffer,
            indices,
            target_q_fn=self._target_q,
            gamma=self.gamma,
            n_step=self.estimation_step,
            rew_norm=False,
        )
        return batch

    def _mse_optimizer(self, batch: Batch, critic: nn.Module, critic_optim: torch.optim.Optimizer) -> Tuple[torch.Tensor, torch.Tensor]:
        current_q = critic(batch.obs, batch.act).flatten()
        target_q = batch.returns.flatten()
        td = target_q - current_q
        critic_loss = (td.pow(2) * getattr(batch, "weight", 1.0)).mean()
        critic_optim.zero_grad()
        critic_loss.backward()
        # Gradient clipping to avoid exploding gradients in critic
        if self.grad_clip_critic_max_norm is not None:
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=self.grad_clip_critic_max_norm)
        critic_optim.step()
        return td.detach(), critic_loss

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        """
        执行算法的核心更新逻辑。
        该方法会计算并缓存梯度信息，使用内部计数器作为 x 轴。
        """
        # +++ 2. 内部学习步数加 1，作为本次更新的 x 轴坐标 +++
        self._learn_step_counter += 1
        current_step = self._learn_step_counter

        # --- Critic 更新 ---
        target_q = batch.returns.flatten()
        gradient_metrics_this_step: Dict[str, Any] = {"step": current_step}

        # Critic 1
        self.critic1_optim.zero_grad()
        current_q1 = self.critic1(batch.obs, batch.act).flatten()
        td1 = target_q - current_q1
        critic1_loss = (td1.pow(2) * getattr(batch, "weight", 1.0)).mean()
        critic1_loss.backward()
        # Gradient clipping for critic1
        if self.grad_clip_critic_max_norm is not None:
            nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=self.grad_clip_critic_max_norm)
        
        # +++ 3. 计算 Critic 1 的梯度并合并到当前步骤的指标字典中 +++
        gradient_metrics_this_step.update(
            calculate_gradient_metrics(self.critic1, "critic1")
        )
        self.critic1_optim.step()

        # Critic 2
        self.critic2_optim.zero_grad()
        current_q2 = self.critic2(batch.obs, batch.act).flatten()
        td2 = target_q - current_q2
        critic2_loss = (td2.pow(2) * getattr(batch, "weight", 1.0)).mean()
        critic2_loss.backward()
        # Gradient clipping for critic2
        if self.grad_clip_critic_max_norm is not None:
            nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=self.grad_clip_critic_max_norm)

        # +++ 4. 计算 Critic 2 的梯度并合并 +++
        gradient_metrics_this_step.update(
            calculate_gradient_metrics(self.critic2, "critic2")
        )
        self.critic2_optim.step()

        batch.weight = (td1.detach() + td2.detach()) / 2.0

        # --- Actor 更新 ---
        obs = batch.obs
        info_cur = getattr(batch, "info", None)
        act, _ = self.actor(obs, state=None, info=info_cur)
        q1 = self.critic1(obs, act).flatten()
        q2 = self.critic2(obs, act).flatten()
        q_min = torch.min(q1, q2)
        
        # (你的 q_min 统计代码保持不变)
        try:
            with torch.no_grad():
                self._min_q_sum_epoch += float(q_min.sum().item())
                self._min_q_count_epoch += int(q_min.numel())
        except Exception:
            pass
            
        entropy = self.actor.action_entropy(batch_size=act.shape[0]).flatten()
        actor_loss = (-q_min - self._alpha * entropy).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        # Gradient clipping for actor
        if self.grad_clip_actor_max_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip_actor_max_norm)

        # +++ 5. 计算 Actor 的梯度并合并 +++
        gradient_metrics_this_step.update(
            calculate_gradient_metrics(self.actor, "actor")
        )
        self.actor_optim.step()
        
        # +++ 6. 将包含所有网络梯度信息的字典追加到缓存中 +++
        self._gradient_metrics_cache.append(gradient_metrics_this_step)

        # --- Alpha 更新 (可选) ---
        if hasattr(self, "_is_auto_alpha") and self._is_auto_alpha:
            log_prob_surrogate = (-entropy).detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_prob_surrogate).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()
        else:
            alpha_loss = None

        self.sync_weight()

        result: Dict[str, float] = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }
        if alpha_loss is not None:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = to_numpy(self._alpha)[()] if isinstance(self._alpha, torch.Tensor) else float(self._alpha)
        return result

    def pop_metrics(self) -> List[Dict[str, Any]]:
        """
        返回缓存的所有梯度指标，并清空缓存。
        这是一个 "pop" 操作，取出后即清空，防止重复记录。
        """
        data = self._gradient_metrics_cache
        self._gradient_metrics_cache = []
        noise = self.actor.pop_noise_stats()
        
        return data

    # ===== Monitoring helpers =====
    def pop_avg_min_q(self) -> Optional[float]:
        """Return the average of collected q_min over the last epoch and reset accumulators.
        If no samples were collected, returns None.
        """
        if self._min_q_count_epoch <= 0:
            return None
        avg = float(self._min_q_sum_epoch / max(1, self._min_q_count_epoch))
        self._min_q_sum_epoch = 0.0
        self._min_q_count_epoch = 0
        return avg

