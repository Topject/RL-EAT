from __future__ import annotations

from typing import Optional, Sequence, Callable

import numpy as np
import torch
import torch.nn as nn

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OffpolicyTrainer

from tianshou.utils.net.common import MLP, Net
from tianshou.utils.net.continuous import Critic


from model.diffusion.rl_diffusion import RLDiffusionActor
from policy.sac_diffusion import SACDDPMPolicy
from env import get_config
from model.attention.self_attention import SelfAttentionQPreprocessNet


class DiffusionMLP(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        device: str = "cpu",
        time_embed_dim: int = 4,
    ):
        super().__init__()
        
        self.time_embed_dim = int(time_embed_dim)
        from tianshou.utils.net.common import MLP  
        self.t_proj = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim), nn.ReLU(inplace=True)
        )
        input_dim = state_dim + action_dim + self.time_embed_dim
        self.mlp = MLP(
            input_dim=input_dim,
            output_dim=action_dim, 
            hidden_sizes=hidden_sizes,
            device=device,
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

        
        combined_input = torch.cat([x_noisy, state, t_embed], dim=1)
        
        return self.mlp(combined_input)



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
    
    critic_hidden_sizes: Sequence[int] = (256, 256),
    
    unet_hidden_sizes: Sequence[int] = (256, 256),
    
    diffusion_timesteps: int = 10,
    time_embed_dim: int = 4,
    
    state_embed_dim: int = 8,
):

    obs_space = train_envs.observation_space[0]
    action_space = train_envs.action_space[0]
    obs_dim = int(np.prod(obs_space.shape))
    action_dim = int(np.prod(action_space.shape))
    max_action = action_space.high[0]

    
    
    
    
    cond_net = Net(state_shape=obs_dim, hidden_sizes=(), device=device).to(device)

    
    unet_model = DiffusionMLP(
        state_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=unet_hidden_sizes,
        time_embed_dim = time_embed_dim,
        device=device,
    ).to(device)

    
    actor = RLDiffusionActor(
        cond_preprocess=cond_net,
        action_dim=action_dim,
        model=unet_model,
        max_action=max_action,
        n_timesteps=diffusion_timesteps,
        beta_schedule='linear',
        sigma_init=0.1,
        action_low=torch.as_tensor(action_space.low, dtype=torch.float32, device=device),
        action_high=torch.as_tensor(action_space.high, dtype=torch.float32, device=device),
    ).to(device)
    

    actor_optim = torch.optim.Adam(actor.parameters(), lr=lr)

    
    _crit_sizes = list(critic_hidden_sizes) if len(critic_hidden_sizes) > 0 else [256]
    first_dim = int(_crit_sizes[0])
    tail_sizes = tuple(_crit_sizes[1:])

    cfg = get_config()
    critic_pre_1 = SelfAttentionQPreprocessNet(
        num_nodes=int(cfg.num_nodes),
        max_visible_tasks=int(cfg.max_visible_tasks),
        action_dim=action_dim,
        node_input_dim=cfg.node_state_dim,
        task_input_dim=cfg.task_state_dim,
        embed_dim=state_embed_dim,
        out_dim=first_dim,
        layer_norm=True,
        dropout_p=0.0,
    ).to(device)
    critic1 = Critic(preprocess_net=critic_pre_1, hidden_sizes=tail_sizes, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=lr)

    critic_pre_2 = SelfAttentionQPreprocessNet(
        num_nodes=int(cfg.num_nodes),
        max_visible_tasks=int(cfg.max_visible_tasks),
        action_dim=action_dim,
        node_input_dim=cfg.node_state_dim,
        task_input_dim=cfg.task_state_dim,
        embed_dim=state_embed_dim,
        out_dim=first_dim,
        layer_norm=True,
        dropout_p=0.0,
    ).to(device)
    critic2 = Critic(preprocess_net=critic_pre_2, hidden_sizes=tail_sizes, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=lr)

    
    fixed_alpha: float = 0.2

    policy = SACDDPMPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        action_space=action_space,
        tau=0.005,
        gamma=0.99,
        alpha=fixed_alpha,
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
