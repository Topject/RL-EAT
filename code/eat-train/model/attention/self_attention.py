from __future__ import annotations

from typing import Optional, Tuple, Dict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionPreprocessNet(nn.Module):
    """
    自注意力预处理网络（替换 CrossAttention 版本的 actor 状态特征提取）。
    - 将 node 向量 (4D) 与 task 向量 (3D) 映射到相同的 state_embed_dim；
    - 将它们作为独立 token 拼接后做单头自注意力；
    - 展平为 (B, (N + L) * D) 的一维特征，供上层 actor 使用。
    接口与 CrossAttentionPreprocessNet 对齐：
      forward(obs, state=None, info=None) -> (feat, state)
      提供 output_dim 供上层网络推断特征维度。
    """
    def __init__(
        self,
        num_nodes: int,
        max_visible_tasks: int,
        node_input_dim: int = 4,
        task_input_dim: int = 3,
        embed_dim: int = 8,
        layer_norm: bool = True,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.max_visible_tasks = int(max_visible_tasks)
        self.node_input_dim = int(node_input_dim)
        self.task_input_dim = int(task_input_dim)
        self.embed_dim = int(embed_dim)

        # token embeds
        self.node_embed = nn.Sequential(
            nn.Linear(self.node_input_dim, self.embed_dim),
            nn.Mish(inplace=True),
        )
        self.task_embed = nn.Sequential(
            nn.Linear(self.task_input_dim, self.embed_dim),
            nn.Mish(inplace=True),
        )

        self.use_ln = bool(layer_norm)
        if self.use_ln:
            self.ln_q = nn.LayerNorm(self.embed_dim)
            self.ln_k = nn.LayerNorm(self.embed_dim)
            self.ln_v = nn.LayerNorm(self.embed_dim)
        else:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()
            self.ln_v = nn.Identity()

        self.dropout = nn.Dropout(p=dropout_p) if dropout_p and dropout_p > 0.0 else nn.Identity()

        # 输出维度 = (N + L) * D
        self.tokens_count = self.num_nodes + self.max_visible_tasks
        self.output_dim = self.tokens_count * self.embed_dim
        self.final_ln = nn.LayerNorm(self.output_dim)

    def _split_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将拼接的一维观测拆分为 nodes 与 tasks 的二维张量。
        期望 obs 形状为 (B, N*4 + L*3)。
        返回：
          nodes: (B, N, 4)
          tasks: (B, L, 3)
        """
        batch_size = obs.shape[0]
        nodes_len = self.num_nodes * self.node_input_dim
        tasks_len = self.max_visible_tasks * self.task_input_dim
        assert obs.shape[1] == nodes_len + tasks_len, (
            f"obs dim mismatch: got {obs.shape[1]}, expect {nodes_len + tasks_len}"
        )
        nodes_flat = obs[:, :nodes_len]
        tasks_flat = obs[:, nodes_len:]
        nodes = nodes_flat.view(batch_size, self.num_nodes, self.node_input_dim)
        tasks = tasks_flat.view(batch_size, self.max_visible_tasks, self.task_input_dim)
        return nodes, tasks

    def forward(
        self,
        obs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        info: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not torch.is_tensor(obs):
            dev = next(self.parameters()).device
            obs = torch.as_tensor(obs, dtype=torch.float32, device=dev)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        if torch.isnan(obs).any():
            raise ValueError("obs is nan")

        nodes, tasks = self._split_obs(obs)

        # (B, N, D), (B, L, D)
        nodes_emb = self.node_embed(nodes)
        tasks_emb = self.task_embed(tasks)

        # 拼接为 (B, N + L, D)
        tokens = torch.cat([nodes_emb, tasks_emb], dim=1)

        # 自注意力
        q = self.ln_q(tokens)
        k = self.ln_k(tokens)
        v = self.ln_v(tokens)
        scale = 1.0 / math.sqrt(self.embed_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, T, T)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)  # (B, T, D)

        feat = context.reshape(context.shape[0], -1)  # (B, T*D)
        feat = self.final_ln(feat)

        if torch.isnan(feat).any():
            raise ValueError("feat is nan")
        return feat, state


class SelfAttentionQPreprocessNet(nn.Module):
    """
    自注意力 Q 预处理网络（替换 CrossAttentionQPreprocessNet）。
    - 将 node 向量 (4D)、task 向量 (3D) 与每个 action 维度 (1D)
      均映射到同一 state_embed_dim；
    - 将所有 token 拼接后做单头自注意力；
    - 展平后经过一个 MLP（与原先 critic 结构保持一致）得到 out_dim 特征；
    - 保持与 Tianshou Critic 兼容的接口与行为。
    提示：若未提供动作，则仅返回状态特征（不经过最终 MLP），与原实现一致。
    """
    def __init__(
        self,
        num_nodes: int,
        max_visible_tasks: int,
        action_dim: int,
        node_input_dim: int = 4,
        task_input_dim: int = 3,
        embed_dim: int = 8,
        out_dim: int = 256,
        layer_norm: bool = True,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.max_visible_tasks = int(max_visible_tasks)
        self.action_dim = int(action_dim)
        self.node_input_dim = int(node_input_dim)
        self.task_input_dim = int(task_input_dim)
        self.embed_dim = int(embed_dim)
        self.output_dim = int(out_dim)

        # token embeds
        self.node_embed = nn.Sequential(
            nn.Linear(self.node_input_dim, self.embed_dim),
            nn.Mish(inplace=True),
        )
        self.task_embed = nn.Sequential(
            nn.Linear(self.task_input_dim, self.embed_dim),
            nn.Mish(inplace=True),
        )
        # 每个动作维度视作一个 1D token
        self.act_embed = nn.Sequential(
            nn.Linear(1, self.embed_dim),
            nn.Mish(inplace=True),
        )

        self.use_ln = bool(layer_norm)
        if self.use_ln:
            self.ln_q = nn.LayerNorm(self.embed_dim)
            self.ln_k = nn.LayerNorm(self.embed_dim)
            self.ln_v = nn.LayerNorm(self.embed_dim)
        else:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()
            self.ln_v = nn.Identity()

        self.dropout = nn.Dropout(p=dropout_p) if dropout_p and dropout_p > 0.0 else nn.Identity()

        # 维度统计
        self.state_tokens = self.num_nodes + self.max_visible_tasks
        self.state_output_dim = self.state_tokens * self.embed_dim
        self.total_tokens = self.state_tokens + self.action_dim

        # 最终 MLP，与原 critic 处理保持一致
        self.fc = nn.Sequential(
            nn.Linear(self.total_tokens * self.embed_dim, self.output_dim),
            nn.Mish(inplace=True),
        )
        # 展平后的归一化层（状态-only 和 状态+动作 各一份，避免维度不匹配）
        self.final_ln_state = nn.LayerNorm(self.state_output_dim)
        self.final_ln_all = nn.LayerNorm(self.total_tokens * self.embed_dim)

    def _split_state_action(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """按约定从 obs 中切分 state 与 action。
        state 期望长度 = N*4 + L*3。若 obs 更长，则剩余部分视作 action。
        返回:
          obs_state: (B, state_len)
          act: (B, action_dim?) 或 None
        """
        state_len = self.num_nodes * self.node_input_dim + self.max_visible_tasks * self.task_input_dim
        total_len = obs.shape[1]
        obs_state = obs[:, :state_len]
        act_tensor: Optional[torch.Tensor] = None
        leftover = total_len - state_len
        if leftover > 0:
            act_tensor = obs[:, state_len:]
        return obs_state, act_tensor

    def _embed_state_tokens(self, obs_state: torch.Tensor) -> torch.Tensor:
        """将拼接的状态向量拆分为 nodes 与 tasks，再分别映射为 token 并拼接。"""
        batch_size = obs_state.shape[0]
        nodes_len = self.num_nodes * self.node_input_dim
        tasks_len = self.max_visible_tasks * self.task_input_dim
        assert obs_state.shape[1] >= nodes_len + tasks_len, (
            f"state dim mismatch: got {obs_state.shape[1]}, expect at least {nodes_len + tasks_len}"
        )
        nodes_flat = obs_state[:, :nodes_len]
        tasks_flat = obs_state[:, nodes_len:nodes_len + tasks_len]
        nodes = nodes_flat.view(batch_size, self.num_nodes, self.node_input_dim)
        tasks = tasks_flat.view(batch_size, self.max_visible_tasks, self.task_input_dim)
        nodes_emb = self.node_embed(nodes)   # (B, N, D)
        tasks_emb = self.task_embed(tasks)   # (B, L, D)
        state_tokens = torch.cat([nodes_emb, tasks_emb], dim=1)  # (B, N+L, D)
        return state_tokens

    def _embed_action_tokens(self, act: torch.Tensor) -> torch.Tensor:
        """将 (B, A) 的动作映射为 (B, A, D) 的 token 表示。"""
        if act.dim() == 1:
            act = act.unsqueeze(0)
        # 保证二维 (B, A)
        act = act.flatten(1)
        act = act.unsqueeze(-1)  # (B, A, 1)
        act_tokens = self.act_embed(act)  # (B, A, D)
        return act_tokens

    def _self_attention(self, tokens: torch.Tensor) -> torch.Tensor:
        """对 (B, T, D) 的 token 做单头自注意力，返回相同形状。"""
        q = self.ln_q(tokens)
        k = self.ln_k(tokens)
        v = self.ln_v(tokens)
        scale = 1.0 / math.sqrt(self.embed_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, T, T)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)  # (B, T, D)
        return context

    def forward(
        self,
        obs: torch.Tensor,
        act: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        info: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        dev = next(self.parameters()).device
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=dev)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        if torch.isnan(obs).any():
            raise ValueError("obs is nan")

        # 拆分 state / action
        obs_state, act_tensor = self._split_state_action(obs)

        # 入参 act 优先
        if act is not None:
            if not torch.is_tensor(act):
                act_tensor = torch.as_tensor(act, dtype=torch.float32, device=dev)
            else:
                act_tensor = act.to(device=dev, dtype=torch.float32)

        # 状态 token
        state_tokens = self._embed_state_tokens(obs_state)  # (B, N+L, D)

        if act_tensor is None:
            # 仅状态路径：返回状态自注意力展平特征（不经过最终 MLP）
            context_s = self._self_attention(state_tokens)  # (B, N+L, D)
            feat_s = context_s.reshape(context_s.shape[0], -1)  # (B, (N+L)*D)
            feat_s = self.final_ln_state(feat_s)
            if torch.isnan(feat_s).any():
                raise ValueError("feat is nan")
            return feat_s, state

        # 拼接状态与动作 token
        act_tokens = self._embed_action_tokens(act_tensor)  # (B, A, D)
        tokens_all = torch.cat([state_tokens, act_tokens], dim=1)  # (B, N+L+A, D)

        # 自注意力 + 展平 + MLP
        context = self._self_attention(tokens_all)  # (B, T, D)
        z = context.reshape(context.shape[0], -1)  # (B, T*D)
        z = self.final_ln_all(z)
        feat = self.fc(z)  # (B, out_dim)

        if torch.isnan(feat).any():
            raise ValueError("feat is nan")
        return feat, state


