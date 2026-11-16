from __future__ import annotations

from typing import List, Optional
import numpy as np
from .backend import Backend, SimulatedBackend, RealBackend
from .node import Node
from .task import SDTask
from .config import EnvConfig


class NodeManager:
    """节点管理器 - 统一管理所有节点，支持仿真/真实模式切换"""
    
    def __init__(
        self,
        config: EnvConfig,
        runtime_config,
        mode: str = "simulated"
    ):
        """初始化节点管理器
        
        Args:
            config: 环境配置
            runtime_config: 运行时配置
            mode: 运行模式 ("simulated" 或 "real")
        """
        self.config = config
        self.runtime_config = runtime_config
        self.mode = mode
        self.num_nodes = config.num_nodes
        
        # 初始化后端
        if mode == "simulated":
            # 仿真模式：创建节点列表
            nodes = self._create_nodes(config)
            self.backend: Backend = SimulatedBackend(nodes)
        elif mode == "real":
            # 真实模式：使用真实后端
            if not runtime_config.node_ips:
                raise ValueError("真实模式需要提供node_ips")
            self.backend: Backend = RealBackend(
                num_nodes=config.num_nodes,
                node_ips=runtime_config.node_ips,
                runtime_config=runtime_config
            )
        else:
            raise ValueError(f"不支持的模式: {mode}")
    
    def _create_nodes(self, config: EnvConfig) -> List[Node]:
        """创建节点列表（仅用于仿真模式）
        
        Args:
            config: 环境配置
            
        Returns:
            节点列表
        """
        nodes = []
        for i in range(config.num_nodes):
            node = Node(
                index=i,
                model_id=0,
                compute_power=config.compute_power[i],
                model_load_seconds=config.model_load_seconds[i],
                model_load_seconds_by_parallelism=getattr(config, "model_load_seconds_by_parallelism", {}),
                time_pred_a=config.time_pred_a,
                time_pred_b=config.time_pred_b,
                linear_parallel_speedup=config.linear_parallel_speedup,
                parallel_speedup_weights=config.parallel_speedup_weights,
                node_type=config.node_types[i],
                network_states=config.network_states,
                network_transition=config.network_transition,
                bandwidth_by_state=config.bandwidth_by_state,
                bandwidth_window_seconds=config.bandwidth_window_seconds,
                remaining_seconds=0.0,
                current_network_state=config.network_states[0],
            )
            nodes.append(node)
        return nodes
    
    def reset(self) -> None:
        """重置所有节点"""
        self.backend.reset()
    
    def get_states(self) -> np.ndarray:
        """获取所有节点的状态
        
        Returns:
            形状为 (num_nodes, state_dim) 的状态数组
        """
        return self.backend.get_all_states()
    
    def get_node_state(self, node_index: int) -> np.ndarray:
        """获取单个节点的状态
        
        Args:
            node_index: 节点索引
            
        Returns:
            节点状态数组
        """
        return self.backend.get_node_state(node_index)
    
    def update(self, dt: float, now: float) -> None:
        """更新所有节点状态
        
        Args:
            dt: 时间增量（秒）
            now: 当前时间戳（秒）
        """
        self.backend.update(dt=dt, now=now)
    
    def assign_task(self, task: SDTask) -> bool:
        """分配任务给节点
        
        Args:
            task: SD任务对象
            
        Returns:
            是否分配成功
        """
        return self.backend.assign_task(task)
    
    def is_node_idle(self, node_index: int) -> bool:
        """检查节点是否空闲
        
        Args:
            node_index: 节点索引
            
        Returns:
            节点是否空闲
        """
        return self.backend.is_node_idle(node_index)
    
    def get_idle_node_indices(self) -> List[int]:
        """获取所有空闲节点的索引
        
        Returns:
            空闲节点索引列表
        """
        return [i for i in range(self.num_nodes) if self.is_node_idle(i)]
    
    def get_node_model_id(self, node_index: int) -> int:
        """获取节点当前加载的模型ID
        
        Args:
            node_index: 节点索引
            
        Returns:
            模型ID（0表示未加载）
        """
        return self.backend.get_node_model_id(node_index)
    
    def set_node_model_id(self, node_index: int, model_id: int) -> None:
        """设置节点的模型ID
        
        Args:
            node_index: 节点索引
            model_id: 模型ID
        """
        self.backend.set_node_model_id(node_index, model_id)
    
    def get_node_remaining_time(self, node_index: int) -> float:
        """获取节点剩余执行时间
        
        Args:
            node_index: 节点索引
            
        Returns:
            剩余时间（秒）
        """
        return self.backend.get_node_remaining_time(node_index)
    
    def is_exact_group(self, chosen: List[int]) -> bool:
        """检查选中的节点是否是一个完整的模型组
        
        Args:
            chosen: 选中的节点索引列表
            
        Returns:
            是否是完整的模型组
        """
        if not chosen:
            return False
        
        group_id = self.get_node_model_id(chosen[0])
        if group_id == 0:
            return False
        
        # 所有选中节点必须有相同的group_id
        for idx in chosen:
            if self.get_node_model_id(idx) != group_id:
                return False
        
        # 且组的大小必须等于选中节点数量
        full_group = [i for i in range(self.num_nodes) if self.get_node_model_id(i) == group_id]
        return sorted(full_group) == sorted(chosen)
    
    def unload_model_groups(self, involved_node_indices: List[int]) -> None:
        """卸载涉及的模型组
        
        Args:
            involved_node_indices: 涉及的节点索引列表
        """
        involved_groups = {
            self.get_node_model_id(i)
            for i in involved_node_indices
            if self.get_node_model_id(i) != 0
        }
        
        # 将所有涉及组的节点model_id设为0
        for i in range(self.num_nodes):
            if self.get_node_model_id(i) in involved_groups:
                self.set_node_model_id(i, 0)
    
    def load_model_group(self, node_indices: List[int], group_id: int) -> None:
        """为节点加载模型组
        
        Args:
            node_indices: 节点索引列表
            group_id: 模型组ID
        """
        for i in node_indices:
            self.set_node_model_id(i, group_id)
    
    def renumber_groups(self) -> None:
        """重新编号模型组，使ID从1开始连续"""
        nonzero_ids = sorted({
            self.get_node_model_id(i)
            for i in range(self.num_nodes)
            if self.get_node_model_id(i) > 0
        })
        
        if not nonzero_ids:
            return
        
        id_map = {old_id: (idx + 1) for idx, old_id in enumerate(nonzero_ids)}
        
        for i in range(self.num_nodes):
            old_id = self.get_node_model_id(i)
            if old_id in id_map:
                self.set_node_model_id(i, id_map[old_id])
    
    def get_max_group_id(self) -> int:
        """获取当前最大的模型组ID
        
        Returns:
            最大模型组ID
        """
        max_id = 0
        for i in range(self.num_nodes):
            model_id = self.get_node_model_id(i)
            if model_id > max_id:
                max_id = model_id
        return max_id

