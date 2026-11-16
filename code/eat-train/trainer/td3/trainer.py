from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import TD3Policy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.continuous import Actor, Critic


class MLPPreprocess(nn.Module):
    """A simple MLP-based preprocess network compatible with Tianshou Actor/Critic.

    It flattens the input from dim=1 and forwards through an MLP. It exposes
    `output_dim` and returns (embedding, state) to match Tianshou's expected API.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        layers = []
        in_dim = int(input_dim)
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_dim, int(hidden)))
            layers.append(nn.ReLU(inplace=True))
            in_dim = int(hidden)
        self.mlp = nn.Sequential(*layers) if len(layers) > 0 else nn.Identity()
        self.output_dim = in_dim

    def forward(self, obs: torch.Tensor, state: Optional[torch.Tensor] = None):
        x = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        x = x.flatten(1)
        x = self.mlp(x)
        return x, state


def get_trainer(
    train_envs: DummyVectorEnv,
    test_envs: Optional[DummyVectorEnv],
    device: str = "cpu",
    lr: float = 3e-4,
    epoch: int = 1000,
    step_per_epoch: int = 10000,
    step_per_collect: int = 1000,
    update_per_step: float = 0.004,
    batch_size: int = 512,
    test_num: int = 2,
    logger=None,
    save_best_fn: Optional[Callable] = None,
    save_checkpoint_fn: Optional[Callable] = None,
    hidden_sizes: Sequence[int] = (256, 256),
):
    """
    获取一个配置好的 OffpolicyTrainer 实例 (TD3)。
    训练逻辑（step-based）：每个 epoch 收集固定步数 step_per_collect，
    然后按 update_per_step * 收集到的步数 进行网络更新。
    """
    obs_space = train_envs.observation_space[0]
    action_space = train_envs.action_space[0]
    obs_dim = int(np.prod(obs_space.shape))
    action_dim = int(np.prod(action_space.shape))
    max_action = action_space.high[0]

    
    actor_pre = MLPPreprocess(input_dim=obs_dim, hidden_sizes=hidden_sizes, device=device).to(device)
    actor = Actor(
        preprocess_net=actor_pre,
        action_shape=action_dim,
        max_action=max_action,
        device=device,
    ).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=lr)

    critic_pre_1 = MLPPreprocess(input_dim=obs_dim + action_dim, hidden_sizes=hidden_sizes, device=device).to(device)
    critic1 = Critic(preprocess_net=critic_pre_1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=lr)

    critic_pre_2 = MLPPreprocess(input_dim=obs_dim + action_dim, hidden_sizes=hidden_sizes, device=device).to(device)
    critic2 = Critic(preprocess_net=critic_pre_2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=lr)

    policy = TD3Policy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        action_space=action_space,
        tau=0.005,
        gamma=0.99,
        exploration_noise=0.1,  
        policy_noise=0.2,
        update_actor_freq=2,
        noise_clip=0.5,
        estimation_step=1,
    ).to(device)

    
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
