from __future__ import annotations

from typing import List
import numpy as np
from .base import Backend
from ..node import Node
from ..task import SDTask


class SimulatedBackend(Backend):
    """仿真后端 - 使用本地时间模拟"""
    
    def __init__(self, nodes: List[Node]):
        """初始化仿真后端
        
        Args:
            nodes: 节点列表
        """
        self.nodes = nodes
    
    def reset(self) -> None:
        """重置所有节点状态"""
        for node in self.nodes:
            node.model_id = 0
            node.remaining_seconds = 0.0
    
    def get_node_state(self, node_index: int) -> np.ndarray:
        """获取单个节点的状态"""
        if node_index < 0 or node_index >= len(self.nodes):
            return np.zeros(2, dtype=np.float32)
        return self.nodes[node_index].state()
    
    def get_all_states(self) -> np.ndarray:
        """获取所有节点的状态"""
        states = [node.state() for node in self.nodes]
        return np.stack(states, axis=0).astype(np.float32)
    
    def assign_task(self, task: SDTask) -> bool:
        """分配任务给节点（仿真模式）"""
        for idx in task.assigned_node_indices:
            if idx >= len(self.nodes):
                return False
            self.nodes[idx].receive(task)
        return True
    
    def update(self, dt: float, now: float) -> None:
        """更新所有节点状态"""
        for node in self.nodes:
            node.update(dt=dt, now=now)
    
    def is_node_idle(self, node_index: int) -> bool:
        """检查节点是否空闲"""
        if node_index < 0 or node_index >= len(self.nodes):
            return False
        return self.nodes[node_index].is_idle()
    
    def get_node_model_id(self, node_index: int) -> int:
        """获取节点当前加载的模型ID"""
        if node_index < 0 or node_index >= len(self.nodes):
            return 0
        return self.nodes[node_index].model_id
    
    def set_node_model_id(self, node_index: int, model_id: int) -> None:
        """设置节点的模型ID"""
        if 0 <= node_index < len(self.nodes):
            self.nodes[node_index].model_id = model_id
    
    def get_node_remaining_time(self, node_index: int) -> float:
        """获取节点剩余执行时间"""
        if node_index < 0 or node_index >= len(self.nodes):
            return 0.0
        return self.nodes[node_index].remaining_seconds

