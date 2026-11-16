from __future__ import annotations

from typing import Optional, Sequence, Callable

import numpy as np
import torch
import torch.nn as nn

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import MLP
from tianshou.utils.net.continuous import Critic

from env import get_config
from config import get_train_config, get_global_config
from model.attention.cross_attention import CrossAttentionPreprocessNet, CrossAttentionQPreprocessNet
from model.attention.self_attention import SelfAttentionQPreprocessNet
from model.diffusion.rl_diffusion_ddim import RLDiffusionActor
from policy.sac_diffusion import SACDDIMPolicy


class DiffusionMLP(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        device: str = "cpu",
        time_embed_dim: int = 4,
    ) -> None:
        super().__init__()
        self.time_embed_dim = int(time_embed_dim)
        from tianshou.utils.net.common import MLP  
        self.t_proj = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim), nn.Mish(inplace=True)
        )
        input_dim = state_dim + action_dim + self.time_embed_dim
        self.mlp = MLP(
            input_dim=input_dim,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            device=device,
            activation=nn.Mish
        )

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        device = x_noisy.device
        half = self.time_embed_dim // 2
        freq = torch.exp(
            torch.arange(half, device=device).float()
            * -(torch.log(torch.tensor(10000.0, device=device)) / max(1, (half - 1)))
        )
        args = t.float().unsqueeze(-1) * freq.unsqueeze(0)
        t_embed = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if t_embed.shape[-1] < self.time_embed_dim:
            t_embed = torch.nn.functional.pad(t_embed, (0, self.time_embed_dim - t_embed.shape[-1]))
        t_embed = self.t_proj(t_embed)
        combined = torch.cat([x_noisy, state, t_embed], dim=1)
        return self.mlp(combined)


def get_trainer(
    train_envs: DummyVectorEnv,
    test_envs: Optional[DummyVectorEnv],
    device: str = "cpu",
    lr: float = 3e-4,
    min_lr: float = None,
    epoch: int = 1000,
    step_per_epoch: int = 10000,
    step_per_collect: int = 1000,
    update_per_step: float = 0.004,
    batch_size: int = 512,
    test_num: int = 2,
    logger = None,
    save_best_fn: Optional[Callable] = None,
    save_checkpoint_fn: Optional[Callable] = None,
    
    actor_hidden_sizes: Sequence[int] = (256, 256),
    critic_hidden_sizes: Sequence[int] = (256, 256),
    unet_hidden_sizes: Sequence[int] = (256, 256),
    
    state_embed_dim: int = 8,
    diffusion_timesteps: int = 5,
    time_embed_dim: int = 4,
    target_entropy = 0.1,
    actor_grad_clip_max_norm: float = None,
    critic_grad_clip_max_norm: float = None,
):
    
    obs_space = train_envs.observation_space[0]
    action_space = train_envs.action_space[0]
    obs_dim = int(np.prod(obs_space.shape))
    action_dim = int(np.prod(action_space.shape))
    max_action = float(action_space.high[0])

    
    cfg = get_config()
    node_numbers = get_global_config().node_num
    attn_pre = CrossAttentionPreprocessNet(
        num_nodes=int(node_numbers),
        max_visible_tasks=int(cfg.max_visible_tasks),
        node_input_dim=cfg.node_state_dim,
        task_input_dim=cfg.task_state_dim,
        embed_dim=state_embed_dim,
        layer_norm=True,
        dropout_p=0.0,
    ).to(device)

    
    unet = DiffusionMLP(
        state_dim=attn_pre.output_dim,
        action_dim=action_dim,
        hidden_sizes=unet_hidden_sizes,
        device=device,
        time_embed_dim=time_embed_dim,
    ).to(device)
    actor = RLDiffusionActor(
        cond_preprocess=attn_pre,
        action_dim=action_dim,
        model=unet,
        max_action=max_action,
        n_timesteps=diffusion_timesteps,
        beta_schedule='linear',
        action_low=torch.as_tensor(action_space.low, dtype=torch.float32, device=device),
        action_high=torch.as_tensor(action_space.high, dtype=torch.float32, device=device),
    ).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim,epoch,min_lr)

    
    _crit = list(critic_hidden_sizes) if len(critic_hidden_sizes) > 0 else [256]
    first_dim = int(_crit[0])
    tail = tuple(_crit[1:])

    cfg_env = get_config()
    critic_pre_1 = SelfAttentionQPreprocessNet(
        num_nodes=int(node_numbers),
        max_visible_tasks=int(cfg_env.max_visible_tasks),
        action_dim=action_dim,
        node_input_dim=cfg.node_state_dim,
        task_input_dim=cfg.task_state_dim,
        embed_dim=state_embed_dim,
        out_dim=first_dim,
        layer_norm=True,
        dropout_p=0.0,
    ).to(device)
    critic1 = Critic(preprocess_net=critic_pre_1, hidden_sizes=critic_hidden_sizes, device=device).to(device)
    
    
    
    
    
    
    
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=lr)

    critic_pre_2 = SelfAttentionQPreprocessNet(
        num_nodes=int(node_numbers),
        max_visible_tasks=int(cfg_env.max_visible_tasks),
        action_dim=action_dim,
        node_input_dim=cfg.node_state_dim,
        task_input_dim=cfg.task_state_dim,
        embed_dim=state_embed_dim,
        out_dim=first_dim,
        layer_norm=True,
        dropout_p=0.0,
    ).to(device)
    critic2 = Critic(preprocess_net=critic_pre_2, hidden_sizes=tail, device=device).to(device)
    
    
    
    
    
    
    
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=lr)

    target_entropy = -np.prod(action_space.shape)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optim = torch.optim.Adam([log_alpha], lr=lr)
    alpha_tuple = (target_entropy, log_alpha, alpha_optim)

    policy = SACDDIMPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        action_space=action_space,
        tau=0.005,
        gamma=0.99,
        alpha=1.0,
        estimation_step=1,
        grad_clip_actor_max_norm=actor_grad_clip_max_norm,
        grad_clip_critic_max_norm=critic_grad_clip_max_norm,
    ).to(device)

    if min_lr:
        actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            actor_optim, T_max=epoch, eta_min=min_lr
        )
        policy.actor_lr_scheduler = actor_scheduler

        critic1_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            critic1_optim, T_max=epoch, eta_min=min_lr
        )
        policy.critic1_lr_scheduler = critic1_scheduler
        
        critic2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            critic2_optim, T_max=epoch, eta_min=min_lr
        )
        policy.critic2_lr_scheduler = critic2_scheduler

    train_collector = Collector(policy, train_envs, VectorReplayBuffer(total_size=step_per_epoch, buffer_num=len(train_envs)))
    test_collector = Collector(policy, test_envs) if test_envs is not None else None

    trainer = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect,
        update_per_step=update_per_step,
        episode_per_test=test_num,
        batch_size=batch_size,
        logger=logger,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        
    )
    return trainer, policy


