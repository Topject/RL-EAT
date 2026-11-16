from __future__ import annotations

from typing import Optional, Sequence, Callable

import numpy as np
import torch
import torch.nn as nn

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import MLP
from tianshou.utils.net.continuous import Critic

from env import get_config
from model.diffusion.rl_diffusion import RLDiffusionActor
from model.attention.cross_attention import CrossAttentionPreprocessNet
from model.diffusion.helpers import SinusoidalPosEmb
from policy.diffusion_ppo import DiffusionPPOPolicy


class DiffusionMLP(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        device: str = "cpu",
        time_embed_dim = 4,
    ):
        super().__init__()
        
        self.time_embed_dim = time_embed_dim
        self.t_embed = SinusoidalPosEmb(self.time_embed_dim)
        input_dim = state_dim + action_dim + self.time_embed_dim
        self.mlp = MLP(
            input_dim=input_dim,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            device=device,
        )

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        
        t_embed = self.t_embed(t.float())
        combined = torch.cat([x_noisy, state, t_embed], dim=1)
        
        
        if torch.isnan(combined).any():
            raise ValueError("combined is nan")
        
        return self.mlp(combined)


def get_trainer(
    train_envs: DummyVectorEnv,
    test_envs: Optional[DummyVectorEnv],
    device: str = "cpu",
    lr: float = 3e-4,
    epoch: int = 1000,
    step_per_epoch: int = 10000,
    repeat_per_collect: int = 8,
    episode_per_collect: int = 8,
    batch_size: int = 512,
    test_num: int = 2,
    logger=None,
    save_best_fn: Optional[Callable] = None,
    save_checkpoint_fn: Optional[Callable] = None,
    critic_hidden_sizes: Sequence[int] = (256, 256),
    unet_hidden_sizes: Sequence[int] = (256, 256),
    state_embed_dim: int = 8,
    diffusion_timesteps: int = 10,
    time_embed_dim: int = 4,
):
    
    action_space = train_envs.action_space[0]
    action_dim = int(np.prod(action_space.shape))
    max_action = float(action_space.high[0])

    
    cfg = get_config()
    num_nodes = int(cfg.num_nodes)
    max_visible_tasks = int(cfg.max_visible_tasks)
    attn_pre = CrossAttentionPreprocessNet(
        num_nodes=num_nodes,
        max_visible_tasks=max_visible_tasks,
        node_input_dim=4,
        task_input_dim=3,
        embed_dim=state_embed_dim,
        layer_norm=True,
        dropout_p=0.0,
    ).to(device)

    
    unet_model = DiffusionMLP(
        state_dim=attn_pre.output_dim,
        action_dim=action_dim,
        hidden_sizes=unet_hidden_sizes,
        device=device,
        time_embed_dim=time_embed_dim,
    ).to(device)
    actor = RLDiffusionActor(
        cond_preprocess=attn_pre,
        action_dim=action_dim,
        model=unet_model,
        max_action=max_action,
        n_timesteps=diffusion_timesteps,
        beta_schedule='linear',
        sigma_init=0.1,
    ).to(device)

    
    critic = Critic(preprocess_net=attn_pre, hidden_sizes=critic_hidden_sizes, device=device).to(device)

    
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

    
    policy = DiffusionPPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        
        action_space=action_space,
        discount_factor=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        vf_coef=0.5,
        ent_coef=0.01,
        
        action_scaling=True,
        deterministic_eval=True,
        recompute_advantage=True,
    ).to(device)

    
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(total_size=step_per_epoch, buffer_num=len(train_envs)),
    )
    test_collector = Collector(policy, test_envs) if test_envs is not None else None

    trainer = OnpolicyTrainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        repeat_per_collect=repeat_per_collect,
        episode_per_collect=episode_per_collect,
        episode_per_test=test_num,
        batch_size=batch_size,
        logger=logger,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
    )
    return trainer, policy
