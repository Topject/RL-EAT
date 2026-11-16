from __future__ import annotations

from typing import Optional, Sequence, Callable

import numpy as np
import torch
import torch.nn as nn

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import MLP, Net
from tianshou.utils.net.continuous import ActorProb, Critic

from env import get_config
from model.diffusion.rl_diffusion import RLDiffusionActor
from model.attention.self_attention import SelfAttentionPreprocessNet
from model.diffusion.helpers import SinusoidalPosEmb
from policy.diffusion_ppo import DiffusionPPOPolicy


class DiffusionMLP(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        device: str = "cpu",
    ):
        super().__init__()
        
        self.time_embed_dim = 32
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
    actor_hidden_sizes: Sequence[int] = (256, 256),
    critic_hidden_sizes: Sequence[int] = (256, 256),
    unet_hidden_sizes: Sequence[int] = (256, 256),
    diffusion_timesteps: int = 10,
):
    
    obs_space = train_envs.observation_space[0]
    obs_dim = int(np.prod(obs_space.shape))
    action_space = train_envs.action_space[0]
    action_dim = int(np.prod(action_space.shape))
    max_action = float(action_space.high[0])

    
    attn_pre = SelfAttentionPreprocessNet(
        obs_dim=obs_dim,
        embed_dim=16,
        layer_norm=True,
        dropout_p=0.0,
    ).to(device)

    
    unet_model = DiffusionMLP(
        state_dim=attn_pre.output_dim,
        action_dim=action_dim,
        hidden_sizes=unet_hidden_sizes,
        device=device,
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
        dist_fn=torch.distributions.Normal,
        action_space=action_space,
        discount_factor=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        vf_coef=0.5,
        ent_coef=0.01,
        reward_normalization=True,
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
