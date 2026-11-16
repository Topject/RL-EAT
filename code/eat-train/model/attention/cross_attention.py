from __future__ import annotations

from typing import Optional, Tuple, Dict, Sequence

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionPreprocessNet(nn.Module):
    """
    将观测中的 node 与 task 向量映射到高维空间，并执行交叉注意力：
    - task 作为 Query
    - node 作为 Key / Value
    输出形状为 (B, max_visible_tasks * embed_dim) 的扁平特征，
    以符合 tianshou Actor/Critic 头部对一维特征的期望。

    注意：该模块遵循 Tianshou preprocess_net 接口：
    forward(obs, state=None, info={}) -> (feat, state)
    并提供 output_dim 属性供 Actor/Critic 推断特征维度。
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

        # Embedding layers
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

        # 输出维度：将 (B, L, D) 展平成 (B, L*D)
        self.output_dim = self.max_visible_tasks * self.embed_dim
        self.final_ln = nn.LayerNorm(self.output_dim)

    def _split_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """将拼接的一维观测拆分为 nodes 与 tasks 的二维张量。
        期望 obs 形状为 (B, N*4 + L*3)。
        返回：
          nodes: (B, N, 4)
          tasks: (B, L, 3)
        """
        batch_size = obs.shape[0]
        nodes_len = self.num_nodes * self.node_input_dim
        tasks_len = self.max_visible_tasks * self.task_input_dim
        assert obs.shape[1] == nodes_len + tasks_len, (
            f"obs dim mismatch: got {obs.shape[1]}, expect {nodes_len + tasks_len}")
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
        # obs: (B, obs_dim) -> split
        if not torch.is_tensor(obs):
            dev = next(self.parameters()).device
            obs = torch.as_tensor(obs, dtype=torch.float32, device=dev)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # check if obs contains nan
        if torch.isnan(obs).any():
            raise ValueError("obs is nan")

        nodes, tasks = self._split_obs(obs)

        # Embedding
        # nodes_emb: (B, N, D), tasks_emb: (B, L, D)
        nodes_emb = self.node_embed(nodes)
        tasks_emb = self.task_embed(tasks)

        # Optional layer norm
        q = self.ln_q(tasks_emb)
        k = self.ln_k(nodes_emb)
        v = self.ln_v(nodes_emb)

        # Cross Attention: attn = softmax(Q K^T / sqrt(D)) V
        # scores: (B, L, N)
        scale = 1.0 / math.sqrt(self.embed_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # context: (B, L, D)
        context = torch.matmul(attn_weights, v)

        # 展平输出
        feat = context.reshape(context.shape[0], -1)
        
        feat = self.final_ln(feat)
        
        # check if feat contains nan
        if torch.isnan(feat).any():
            raise ValueError("feat is nan")
        
        return feat, state


class CrossAttentionQPreprocessNet_(nn.Module):
    """
    针对 Critic 的预处理网络：
    - 先做与 CrossAttentionPreprocessNet 相同的交叉注意力，得到 obs 特征 f_s
    - 与动作向量拼接 [f_s, a]，通过线性映射得到固定维度的特征
    - 提供 output_dim，供 Critic 上层线性头使用
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
        self.obs_net = CrossAttentionPreprocessNet(
            num_nodes=num_nodes,
            max_visible_tasks=max_visible_tasks,
            node_input_dim=node_input_dim,
            task_input_dim=task_input_dim,
            embed_dim=embed_dim,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )
        self.action_dim = int(action_dim)
        self.output_dim = int(out_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.obs_net.output_dim + self.action_dim, self.output_dim),
            nn.Mish(inplace=True),
        )

    def forward(
        self,
        obs: torch.Tensor,
        act: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        info: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # obs 可能是 [state] 或 [state, action]，需要自适应拆分
        dev = next(self.parameters()).device
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=dev)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # 计算期望的状态长度与动作长度
        state_len = (
            self.obs_net.num_nodes * self.obs_net.node_input_dim
            + self.obs_net.max_visible_tasks * self.obs_net.task_input_dim
        )
        total_len = obs.shape[1]

        # 提取状态与动作（若拼接在一起则从 obs 中切分）
        obs_state = obs
        act_tensor: Optional[torch.Tensor] = None
        if total_len == state_len:
            # 只有状态（通常用于 actor 或某些 target 路径）
            pass
        elif total_len >= state_len + self.action_dim:
            # critic 场景：最后 action_dim 视为动作
            obs_state = obs[:, :state_len]
            act_tensor = obs[:, state_len:state_len + self.action_dim]
        elif total_len > state_len:
            # 兼容：将剩余部分都当作动作
            obs_state = obs[:, :state_len]
            act_tensor = obs[:, state_len:]

        # 通过交叉注意力提取状态特征
        feat_s, _ = self.obs_net(obs_state, state=None, info=info)

        # 若函数入参提供了 act，则优先使用
        if act is not None:
            if not torch.is_tensor(act):
                act_tensor = torch.as_tensor(act, dtype=torch.float32, device=dev)
            else:
                act_tensor = act.to(device=dev, dtype=torch.float32)

        # 如果仍无动作（例如某些 target 路径），仅返回状态特征
        if act_tensor is None:
            return feat_s, state

        if act_tensor.dim() == 1:
            act_tensor = act_tensor.unsqueeze(0)
        # 保证二维 (B, action_dim)
        act_tensor = act_tensor.flatten(1)

        x = torch.cat([feat_s, act_tensor], dim=1)
        feat = self.fc(x)
        return feat, state


class CrossAttentionQPreprocessNet_(nn.Module):
    """
    针对 Critic 的预处理网络：
    - 先做与 CrossAttentionPreprocessNet 相同的交叉注意力，得到 obs 特征 f_s
    - 与动作向量拼接 [f_s, a]，通过线性映射得到固定维度的特征
    - 提供 output_dim，供 Critic 上层线性头使用
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
        self.obs_net = CrossAttentionPreprocessNet(
            num_nodes=num_nodes,
            max_visible_tasks=max_visible_tasks,
            node_input_dim=node_input_dim,
            task_input_dim=task_input_dim,
            embed_dim=embed_dim,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )
        self.action_dim = int(action_dim)
        self.output_dim = int(out_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.obs_net.output_dim + self.action_dim, self.output_dim),
            nn.Mish(inplace=True),
        )

    def forward(
        self,
        obs: torch.Tensor,
        act: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        info: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # obs 可能是 [state] 或 [state, action]，需要自适应拆分
        dev = next(self.parameters()).device
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=dev)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # 计算期望的状态长度与动作长度
        state_len = (
            self.obs_net.num_nodes * self.obs_net.node_input_dim
            + self.obs_net.max_visible_tasks * self.obs_net.task_input_dim
        )
        total_len = obs.shape[1]

        # 提取状态与动作（若拼接在一起则从 obs 中切分）
        obs_state = obs
        act_tensor: Optional[torch.Tensor] = None
        if total_len == state_len:
            # 只有状态（通常用于 actor 或某些 target 路径）
            pass
        elif total_len >= state_len + self.action_dim:
            # critic 场景：最后 action_dim 视为动作
            obs_state = obs[:, :state_len]
            act_tensor = obs[:, state_len:state_len + self.action_dim]
        elif total_len > state_len:
            # 兼容：将剩余部分都当作动作
            obs_state = obs[:, :state_len]
            act_tensor = obs[:, state_len:]

        # 通过交叉注意力提取状态特征
        feat_s, _ = self.obs_net(obs_state, state=None, info=info)

        # 若函数入参提供了 act，则优先使用
        if act is not None:
            if not torch.is_tensor(act):
                act_tensor = torch.as_tensor(act, dtype=torch.float32, device=dev)
            else:
                act_tensor = act.to(device=dev, dtype=torch.float32)

        # 如果仍无动作（例如某些 target 路径），仅返回状态特征
        if act_tensor is None:
            return feat_s, state

        if act_tensor.dim() == 1:
            act_tensor = act_tensor.unsqueeze(0)
        # 保证二维 (B, action_dim)
        act_tensor = act_tensor.flatten(1)

        x = torch.cat([feat_s, act_tensor], dim=1)
        feat = self.fc(x)
        return feat, state

class CrossAttentionQPreprocessNet(nn.Module):
    """
    针对 Critic 的预处理网络:
    - 状态 s 经过交叉注意力网络得到高阶特征 feat_s
    - 动作 a 经过一个独立的线性嵌入层得到特征 feat_a
    - 拼接 [feat_s, feat_a]，通过最终的 MLP 得到 Q 值的前驱特征
    - 这种结构确保了在拼接时，状态和动作的特征处于相似的抽象层次和数值尺度。
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
        layer_norm: bool = False,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        # 1. 状态处理网络 (和原来一样)
        self.obs_net = CrossAttentionPreprocessNet(
            num_nodes=num_nodes,
            max_visible_tasks=max_visible_tasks,
            node_input_dim=node_input_dim,
            task_input_dim=task_input_dim,
            embed_dim=embed_dim,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )
        self.action_dim = int(action_dim)
        self.output_dim = int(out_dim)

        # 2. === 新增：动作嵌入层 ===
        # 将动作向量映射到与状态特征更匹配的维度空间
        # 一个简单的线性层 + 激活函数即可
        action_embed_size = 4
        self.action_embed = nn.Sequential(
            nn.Linear(self.action_dim, action_embed_size),
            nn.ReLU(inplace=True)
        )

        # 3. 最终的合并网络
        # 输入维度是状态特征维度 + 动作嵌入维度
        self.fc = nn.Sequential(
            nn.Linear(self.obs_net.output_dim + action_embed_size, self.output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        obs: torch.Tensor,
        act: Optional[torch.Tensor] = None, # 这个参数在Tianshou的Critic中不会被直接使用
        state: Optional[torch.Tensor] = None,
        info: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        dev = next(self.parameters()).device
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=dev)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # Tianshou 的 Critic 会将 state 和 action 拼接后作为 obs 传入
        state_len = self.obs_net.num_nodes * self.obs_net.node_input_dim + \
                    self.obs_net.max_visible_tasks * self.obs_net.task_input_dim
        
        # 从 obs 中拆分出状态和动作
        obs_state = obs[:, :state_len]
        act_tensor = obs[:, state_len:]
        
        # 1. 处理状态 -> 得到高阶状态特征 feat_s
        feat_s, _ = self.obs_net(obs_state, state=None, info=info)

        # 2. === 新增：处理动作 -> 得到高阶动作特征 feat_a ===
        feat_a = self.action_embed(act_tensor)

        # 3. 拼接高阶特征
        x = torch.cat([feat_s, feat_a], dim=1)
        
        # 4. 通过最终网络
        feat = self.fc(x)
        return feat, state

class MLPCritic(nn.Module):
    """
    一个简单的MLP预处理网络，用于替代复杂的CrossAttentionQPreprocessNet。
    
    它接收Tianshou Critic传入的拼接后的 (state, action) 张量，
    并将其通过一个标准的多层感知机（MLP）进行处理。
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        device: str = "cpu",
    ) -> None:
        super().__init__()
        from tianshou.utils.net.common import MLP
        
        # 输入维度是扁平化的观测维度 + 动作维度
        self.input_dim = obs_dim + action_dim
        
        # Tianshou 的 Critic 会自动在最后添加一个输出层（将特征映射到标量Q值），
        # 所以这里的 MLP 只需要输出最后一层隐藏层的特征即可。
        # 因此，output_dim 设置为 0，让 Tianshou 的 MLP 自动推断。
        self.model = MLP(
            input_dim=self.input_dim,
            hidden_sizes=hidden_sizes,
            device=device,
            # 使用 ReLU 作为激活函数，它通常比 Mish 更稳定，是更安全的选择
            activation=nn.ReLU, 
        )
        # self.output_dim 会被 Tianshou 的 MLP 自动设置
        self.output_dim = self.model.output_dim

    def forward(
        self,
        obs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        info: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        obs 是 Tianshou 传入的 torch.cat([state, action], dim=1)
        """
        # 直接将拼接后的 obs 输入 MLP
        features = self.model(obs)
        return features, state
