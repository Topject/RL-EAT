from __future__ import annotations

from typing import Optional, Sequence, Callable

import numpy as np
import torch

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.continuous import ActorProb, Critic

from env import get_config
from model.attention.cross_attention import CrossAttentionPreprocessNet


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
    embed_dim: int = 16,
    actor_hidden_sizes: Sequence[int] = (256, 256),
    critic_hidden_sizes: Sequence[int] = (256, 256),
):
    
    obs_space = train_envs.observation_space[0]
    action_space = train_envs.action_space[0]
    action_dim = int(np.prod(action_space.shape))
    max_action = float(action_space.high[0])

    
    cfg = get_config()
    num_nodes = int(cfg.num_nodes)
    max_visible_tasks = int(cfg.max_visible_tasks)

    
    obs_pre = CrossAttentionPreprocessNet(
        num_nodes=num_nodes,
        max_visible_tasks=max_visible_tasks,
        node_input_dim=4,
        task_input_dim=3,
        embed_dim=embed_dim,
        layer_norm=True,
        dropout_p=0.0,
    ).to(device)

    actor = ActorProb(
        preprocess_net=obs_pre,
        action_shape=action_dim,
        max_action=max_action,
        device=device,
        unbounded=True,
        hidden_sizes=actor_hidden_sizes,
    ).to(device)
    critic = Critic(preprocess_net=obs_pre, hidden_sizes=critic_hidden_sizes, device=device).to(device)

    
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

    
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


