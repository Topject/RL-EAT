from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Dict, List, Optional


# ============= 全局实验配置 =============

@dataclass
class GlobalConfig:
    """全局实验配置
    
    这是实验的核心配置，包含：
    1. 节点规模配置
    2. 任务队列可见长度
    
    根据节点数量自动配置：
    - 任务到达率 (Poisson lambda): N/32 任务/秒
      * 16节点: λ=0.5, 32节点: λ=1.0, 48节点: λ=1.5, 64节点: λ=2.0
    - 节点异构算力: 4种类型均匀分布 (0.5x, 1.0x, 1.5x, 2.0x)
    - 模型加载时间: 统一为 0.2 秒
    """
    # ===== 需要手动设置的参数 =====
    node_num: int = 32              # 节点数量 (16/32/48/64)
    max_visible_tasks: int = 8      # 最大可见任务数量 l
    
    # ===== 自动计算的参数 =====
    @property
    def poisson_lambda(self) -> float:
        """根据节点数量计算任务到达率"""

        if self.node_num == 32:
            return 0.3
        else:
            # 线性插值
            return 0.3 * self.node_num / 32.0
    
    @property
    def description(self) -> str:
        """实验描述"""
        scale_map = {16: "小规模", 32: "中等规模", 48: "大规模", 64: "超大规模"}
        scale_name = scale_map.get(self.node_num, f"{self.node_num}节点")
        return f"{scale_name}集群：{self.node_num}节点，λ={self.poisson_lambda}任务/秒"


# ============= 训练配置 =============

@dataclass
class TrainConfig:
    """训练超参数配置"""
    seed: int = 42
    train_num: int = 8              # 并行训练环境数量
    test_num: int = 4               # 并行测试环境数量
    save_interval_epochs: int = 500 # checkpoint保存间隔
    
    # 网络结构
    actor_hidden_sizes: Sequence[int] = (256, 256)
    critic_hidden_sizes: Sequence[int] = (256, 256)
    unet_hidden_size: Sequence[int] = (256, 256)
    
    # Attention 层嵌入维度
    state_embed_dim: int = 8
    
    # Diffusion 动作头参数
    diffusion_steps: int = 4
    diffusion_time_embed_dim: int = 8
    diffusion_sigma_init: float = 0.5
    diffusion_beta_schedule: str = "linear"
    ddim_target_entropy: float = 0.1
    
    # 梯度裁剪
    grad_clip_actor_max_norm: float = 0.5
    grad_clip_critic_max_norm: float = 1.0


# ============= 运行时配置 =============

@dataclass
class RuntimeConfig:
    """运行时配置 - 控制仿真/真实环境切换"""
    mode: str = "simulated"  # "simulated" or "real"
    node_ips: List[str] = field(default_factory=list)  # 节点IP地址列表（真实模式）
    
    # 真实环境分布式参数
    master_addr: str = "localhost"
    master_port: int = 29500
    
    # API参数
    api_timeout: float = 30.0
    output_dir: str = "outputs"
    debug: bool = False  # 是否输出调试信息
    
    # 真实环境的默认参数（对应api.json中的init参数）
    height: int = 512
    width: int = 512
    warmup_steps: int = 4
    backend_mode: str = "corrected_async_gn"
    parallelism: str = "patch"
    split_batch: bool = True
    use_cuda_graph: bool = True
    guidance_scale: float = 5.0
    
    # 绘图参数
    num_inference_steps: int = 50
    seed: int = 1234


# ============= 配置获取函数 =============

_global_config = GlobalConfig()
_train_config = TrainConfig()
_runtime_config = RuntimeConfig()


def get_global_config() -> GlobalConfig:
    """获取全局配置"""
    return _global_config


def get_train_config() -> TrainConfig:
    """获取训练配置"""
    return _train_config


def get_runtime_config() -> RuntimeConfig:
    """获取运行时配置"""
    return _runtime_config


def set_global_config(node_num: int = 48, max_visible_tasks: int = 8) -> None:
    """设置全局配置
    
    Args:
        node_num: 节点数量 (16/32/48/64)
        max_visible_tasks: 最大可见任务数量
    """
    global _global_config
    _global_config = GlobalConfig(
        node_num=node_num,
        max_visible_tasks=max_visible_tasks
    )
    print(f"实验配置: {_global_config.description}")
    print(f"  - 节点数量: {_global_config.node_num}")
    print(f"  - 任务到达率: {_global_config.poisson_lambda} 任务/秒")
    print(f"  - 最大可见任务: {_global_config.max_visible_tasks}")


def set_runtime_config(
    mode: str = "simulated",
    node_ips: Optional[List[str]] = None,
    master_addr: str = "localhost",
    master_port: int = 29500,
    output_dir: str = "outputs",
    debug: bool = False,
    **kwargs
) -> None:
    """设置运行时配置
    
    Args:
        mode: 运行模式 ("simulated" 或 "real")
        node_ips: 节点IP地址列表（真实模式必需）
        master_addr: 分布式训练主节点地址
        master_port: 分布式训练端口
        output_dir: 输出目录
        debug: 是否输出调试信息
        **kwargs: 其他运行时参数
    """
    global _runtime_config
    _runtime_config = RuntimeConfig(
        mode=mode,
        node_ips=node_ips or [],
        master_addr=master_addr,
        master_port=master_port,
        output_dir=output_dir,
        debug=debug,
        **kwargs
    )
    print(f"运行时配置: 模式={_runtime_config.mode}")
    if mode == "real":
        print(f"  - 节点数量: {len(_runtime_config.node_ips)}")
        print(f"  - Master地址: {_runtime_config.master_addr}:{_runtime_config.master_port}")