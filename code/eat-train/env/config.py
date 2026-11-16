from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict


@dataclass
class EnvConfig:
    """环境配置
    
    大部分参数从 GlobalConfig 自动获取，也可以手动覆盖。
    """
    # ===== 核心参数（从 GlobalConfig 读取，也可手动覆盖）=====
    max_visible_tasks: int = 8      # 最大可见任务数量 l
    num_nodes: int = 32             # 节点数量
    poisson_lambda_per_sec: float = 1.0  # 任务到达率（会在 __post_init__ 中自动设置）
    
    # ===== 环境基础参数 =====
    max_step: int = 30              # 最大步数选项
    concat_state: bool = False      # 是否返回拼接的观察向量
    
    # 时间控制
    dt_seconds: float = 0.5         # 时间步长
    max_sim_seconds: float = 300.0  # 最大模拟时间
    increment_time_on_legal_action: bool = True
    
    # 任务生成
    total_tasks: int = 32         # 总任务数
    prompt_pool: List[str] = field(
        default_factory=lambda: [
            "a cute cat, watercolor, 4k",
            "sunset over mountains, volumetric light, ultra-detailed",
            "futuristic city skyline at night, neon lights",
            "portrait of a medieval knight, dramatic lighting",
            "a serene beach, tropical, film grain",
        ]
    )
    # parallel_choices: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    parallel_choices: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    
    # ===== 节点配置 =====
    # 统一模型加载时间（秒）
    model_load_seconds_unified: float = 0
    model_load_seconds: List[float] = field(default_factory=list)
    model_load_seconds_by_parallelism: Dict[int, float] = field(default_factory=dict)
    
    # 异构算力配置：4种节点类型 (0.5x, 1.0x, 1.5x, 2.0x)
    compute_power_types: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    compute_power: List[float] = field(default_factory=list)
    
    # 节点类型（简化，均为 cluster）
    node_types: List[str] = field(default_factory=list)
    
    # ===== 网络参数（保留以兼容，但不关键）=====
    network_states: Tuple[str, str] = ("good", "normal")
    network_transition: Dict[str, List[List[float]]] = field(
        default_factory=lambda: {
            "edge": [[0.95, 0.05], [0.10, 0.90]],
            "cluster": [[0.98, 0.02], [0.05, 0.95]],
        }
    )
    bandwidth_by_state: Dict[str, float] = field(
        default_factory=lambda: {"good": 100.0, "normal": 100.0}
    )
    bandwidth_window_seconds: float = 10.0
    
    # ===== 执行时间预测 =====
    # 执行时间 = time_pred_a * steps + time_pred_b (秒)
    time_pred_a: float = 1.0
    time_pred_b: float = 0.0
    
    # 并行加速权重（实测值）
    linear_parallel_speedup: bool = True
    parallel_speedup_weights: Dict[int, float] = field(
        default_factory=lambda: {
            1: 0.39984006397441,
            2: 0.723589001447178,
            4: 1.06044538706257,
            8: 1.51975683890577
        }
    )
    
    # ===== 奖励函数参数 =====
    reward_t1_scale: float = 50     # t1 归一化尺度 平均执行时间 35s
    reward_t1_weight: float = 3     # t1 权重
    reward_t2_scale: float = 50      # t2 归一化尺度
    reward_t2_weight: float = 1     # t2 权重
    reward_q_scale: float = 30      # quality 归一化尺度
    reward_q_weight: float = 6      # quality 权重
    reward_bais: float = 5          # 奖励偏置

    # ===== 状态维度 =====
    node_state_dim: int = 2         # 节点状态维度
    task_state_dim: int = 2         # 任务状态维度
    
    # ===== 其他参数 =====
    # 镜像大小（MB），用于传输时间估算。设为 0 忽略传输时间
    image_size_mb: float = 0.0
    
    # 统计
    enable_statistics: bool = False
    stats_csv_path: str = "env_stats.csv"
    
    def __post_init__(self):
        """初始化后自动配置节点相关参数"""
        n = int(self.num_nodes)
        
        # 1. 根据节点数量自动设置任务到达率
        print("setting poisson lambda per sec")
        if n == 32:
            self.poisson_lambda_per_sec = 0.3
        else:
            # 线性插值
            self.poisson_lambda_per_sec = 0.3 * n / 32.0
        
        # 2. 设置统一模型加载时间
        self.model_load_seconds = [self.model_load_seconds_unified] * n
        if not self.model_load_seconds_by_parallelism:
            for p in [1, 2, 4, 8, 16, 32]:
                self.model_load_seconds_by_parallelism[p] = self.model_load_seconds_unified
        
        # 3. 配置异构算力：节点均匀分配到4种类型
        if not self.compute_power:
            num_types = len(self.compute_power_types)
            self.compute_power = []
            for i in range(n):
                type_idx = i % num_types
                self.compute_power.append(self.compute_power_types[type_idx])
        else:
            # 调整 compute_power 列表长度
            if len(self.compute_power) < n:
                fill_value = self.compute_power[-1] if self.compute_power else 1.0
                self.compute_power += [fill_value] * (n - len(self.compute_power))
            elif len(self.compute_power) > n:
                self.compute_power = self.compute_power[:n]
        
        # 4. 配置节点类型（简化，均为 cluster）
        if not self.node_types:
            self.node_types = ["cluster"] * n
        else:
            if len(self.node_types) < n:
                fill_value = self.node_types[-1] if self.node_types else "cluster"
                self.node_types += [fill_value] * (n - len(self.node_types))
            elif len(self.node_types) > n:
                self.node_types = self.node_types[:n]


def get_config(num_nodes: int = None, max_visible_tasks: int = None) -> EnvConfig:
    """获取环境配置
    
    Args:
        num_nodes: 节点数量，如果不指定则从 GlobalConfig 读取
        max_visible_tasks: 最大可见任务数，如果不指定则从 GlobalConfig 读取
    
    Returns:
        EnvConfig 实例
    """
    # 尝试从主配置读取
    try:
        from config import get_global_config
        global_cfg = get_global_config()
        if num_nodes is None:
            num_nodes = global_cfg.node_num
        if max_visible_tasks is None:
            max_visible_tasks = global_cfg.max_visible_tasks
    except ImportError:
        # 如果无法导入，使用默认值
        if num_nodes is None:
            num_nodes = 32
        if max_visible_tasks is None:
            max_visible_tasks = 8
    
    num_nodes = 64
    return EnvConfig(
        num_nodes=num_nodes,
        max_visible_tasks=max_visible_tasks
    )