from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from ..task import SDTask


class Backend(ABC):
    """后端抽象基类 - 定义仿真和真实环境的统一接口"""
    
    @abstractmethod
    def reset(self) -> None:
        """重置后端状态"""
        pass
    
    @abstractmethod
    def get_node_state(self, node_index: int) -> np.ndarray:
        """获取单个节点的状态
        
        Args:
            node_index: 节点索引
            
        Returns:
            节点状态数组 [remaining_time, model_id, ...]
        """
        pass
    
    @abstractmethod
    def get_all_states(self) -> np.ndarray:
        """获取所有节点的状态
        
        Returns:
            形状为 (num_nodes, state_dim) 的状态数组
        """
        pass
    
    @abstractmethod
    def assign_task(self, task: SDTask) -> bool:
        """分配任务给节点
        
        Args:
            task: SD任务对象
            
        Returns:
            是否分配成功
        """
        pass
    
    @abstractmethod
    def update(self, dt: float, now: float) -> None:
        """更新后端状态
        
        Args:
            dt: 时间增量（秒）
            now: 当前时间戳（秒）
        """
        pass
    
    @abstractmethod
    def is_node_idle(self, node_index: int) -> bool:
        """检查节点是否空闲
        
        Args:
            node_index: 节点索引
            
        Returns:
            节点是否空闲
        """
        pass
    
    @abstractmethod
    def get_node_model_id(self, node_index: int) -> int:
        """获取节点当前加载的模型ID
        
        Args:
            node_index: 节点索引
            
        Returns:
            模型ID（0表示未加载）
        """
        pass
    
    @abstractmethod
    def set_node_model_id(self, node_index: int, model_id: int) -> None:
        """设置节点的模型ID
        
        Args:
            node_index: 节点索引
            model_id: 模型ID
        """
        pass
    
    @abstractmethod
    def get_node_remaining_time(self, node_index: int) -> float:
        """获取节点剩余执行时间
        
        Args:
            node_index: 节点索引
            
        Returns:
            剩余时间（秒）
        """
        pass

