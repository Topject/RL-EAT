from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

def get_trainer(
    train_envs: DummyVectorEnv,
    test_envs: Optional[DummyVectorEnv],
    device: str = "cpu",
    lr: float = 3e-4,
    epoch: int = 10,
    step_per_epoch: int = 10000,
    repeat_per_collect: int = 8,
    episode_per_collect: int = 8,
    batch_size: int = 512,
    test_num: int = 1,
    logger=None,
    save_best_fn: Optional[Callable] = None,
    save_checkpoint_fn: Optional[Callable] = None,
    actor_hidden_sizes: Sequence[int] = (256, 256),
    critic_hidden_sizes: Sequence[int] = (256, 256),
):

    
    obs_space = train_envs.observation_space[0]
    action_space = train_envs.action_space[0]
    obs_dim = int(np.prod(obs_space.shape))
    action_dim = int(np.prod(action_space.shape))
    
    max_action = action_space.high[0]

    net = Net(
        state_shape=obs_dim,
        hidden_sizes=actor_hidden_sizes,
        device=device
    )
    
    actor = ActorProb(
        preprocess_net=net,
        action_shape=action_dim,
        max_action=max_action,
        unbounded=True,  
        device=device,
    ).to(device)
    
    critic = Critic(
        preprocess_net=net,
        hidden_sizes=critic_hidden_sizes,
        device=device
    ).to(device)

    actor_critic_params = list(actor.parameters()) + list(critic.parameters())
    optim = torch.optim.Adam(actor_critic_params, lr=lr)

    policy = PPOPolicy(
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
    )

    policy.to(device)

    
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(total_size=step_per_epoch, buffer_num=len(train_envs)))
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