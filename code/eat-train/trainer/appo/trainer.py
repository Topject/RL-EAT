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
from model.attention.self_attention import SelfAttentionPreprocessNet


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
    """
    使用自注意力作为特征抽取（preprocess_net）的 PPO 训练器。

    - 输入为拼接后的整体状态向量（不区分 node 与 task）
    - 将每个标量 token 线性映射至维度 d=embed_dim，做自注意力
    - 输出展平为一维特征，作为 Actor/Critic 的输入
    """
    # 读取空间信息
    obs_space = train_envs.observation_space[0]
    obs_dim = int(np.prod(obs_space.shape))
    action_space = train_envs.action_space[0]
    action_dim = int(np.prod(action_space.shape))
    max_action = float(action_space.high[0])

    # 自注意力预处理网络（基于整体状态向量），Actor/Critic 共享
    obs_pre = SelfAttentionPreprocessNet(
        obs_dim=obs_dim,
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

    # 共享优化器（Actor + Critic）
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

    # PPO 策略
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

    # Collector 与 Trainer
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


