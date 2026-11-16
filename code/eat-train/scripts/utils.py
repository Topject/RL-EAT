import os
import csv
import json
import fcntl
import pynvml
import numpy as np
import logging
import torch.nn as nn
from typing import Dict, List, Optional
from datetime import datetime

def calculate_gradient_metrics(model: nn.Module, tag_prefix: str) -> Dict[str, float]:
    """
    计算模型参数的梯度指标（L2范数和稀疏性），并返回一个字典。

    Args:
        model (nn.Module): 需要检查梯度的模型。
        tag_prefix (str): 用于生成字典键的前缀 (e.g., 'gradients_norm/actor')。

    Returns:
        Dict[str, float]: 包含梯度指标的字典。
    """
    metrics: Dict[str, float] = {}
    total_norm_sq = 0.0
    
    for name, p in model.named_parameters():
        if p.grad is not None:
            # 1. 梯度L2范数 (Gradient L2 Norm)
            param_norm = p.grad.detach().data.norm(2).item()
            metrics[f"{tag_prefix}_norm/{name}"] = param_norm
            total_norm_sq += param_norm ** 2

            # 2. 梯度稀疏性 (Gradient Sparsity)
            num_zeros = (p.grad == 0).sum().item()
            total_elements = p.grad.numel()
            sparsity = num_zeros / total_elements
            metrics[f"{tag_prefix}_sparsity/{name}"] = sparsity

    total_norm = total_norm_sq ** 0.5
    metrics[f"{tag_prefix}_total_norm"] = total_norm
    
    return metrics

def set_best_gpu():
    """
    自动检测并设置CUDA_VISIBLE_DEVICES为当前负载最小的GPU。
    只考虑GPU的占用率（utilization）。
    """
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            logging.warning("系统中未检测到NVIDIA GPU。")
            return

        gpu_utilizations = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            # 获取GPU占用率信息
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            gpu_utilizations.append(utilization)
            logging.info(f"GPU {i} 的当前占用率: {utilization}%")

        # 找到占用率最低的GPU索引
        best_gpu_id = np.argmin(gpu_utilizations)
        min_utilization = gpu_utilizations[best_gpu_id]
        
        logging.info(f"检测到负载最小的GPU是 {best_gpu_id}，占用率为 {min_utilization}%。")

        # 设置环境变量
        os.environ['CUDA_VISIBLE_DEVICES'] = str(best_gpu_id)
        logging.info(f"已设置环境变量 CUDA_VISIBLE_DEVICES = {best_gpu_id}")

    except pynvml.NVMLError as error:
        logging.error(f"访问NVIDIA GPU失败: {error}")
    finally:
        # 确保在程序结束时关闭pynvml
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


def save_eval_details(ckpt_dir: str, all_results: List[Dict]) -> str:
    """
    保存每次评估的详细结果到CSV文件。
    
    Args:
        ckpt_dir: checkpoint目录路径
        all_results: 包含所有评估结果的列表，每个元素是一个字典
    
    Returns:
        保存的CSV文件路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_csv = os.path.join(ckpt_dir, f"eval_details_{timestamp}.csv")
    
    with open(details_csv, 'w', newline='') as f:
        fieldnames = ['run_id', 'seed', 'total_reward', 'completed_count', 'avg_resp', 'avg_quality', 'steps']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    
    print(f"\n详细结果已保存到: {details_csv}")
    return details_csv


def save_eval_summary(ckpt_dir: str, repeat_times: int, avg_reward: float, 
                      avg_completed: float, avg_resp_time: float, 
                      avg_quality: float, avg_steps: float) -> str:
    """
    保存平均评估结果到单独的CSV文件。
    
    Args:
        ckpt_dir: checkpoint目录路径
        repeat_times: 重复评估次数
        avg_reward: 平均总奖励
        avg_completed: 平均完成数量
        avg_resp_time: 平均响应时间
        avg_quality: 平均质量
        avg_steps: 平均步数
    
    Returns:
        保存的CSV文件路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = os.path.join(ckpt_dir, f"eval_summary_{timestamp}.csv")
    
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['repeat_times', repeat_times])
        writer.writerow(['avg_total_reward', f"{avg_reward:.4f}"])
        writer.writerow(['avg_completed_count', f"{avg_completed:.2f}"])
        writer.writerow(['avg_response_time', f"{avg_resp_time:.4f}"])
        writer.writerow(['avg_quality', f"{avg_quality:.4f}"])
        writer.writerow(['avg_steps', f"{avg_steps:.2f}"])
    
    print(f"平均结果已保存到: {summary_csv}")
    return summary_csv


def find_latest_ckpt(node_num: int, quality_ratio: float, algo_prefix: str, runs_root: str = "runs") -> str:
    """
    在 runs 目录下根据节点数、质量系数与算法前缀寻找最新的 checkpoint 文件。

    目录结构假设：
      runs/{node_num}_{quality_ratio:.3f}/{algo_dir}/YYYY-mm-dd_HH-MM-SS/
    其中 {algo_dir} 以 algo_prefix 开头，例如：
      - ppo
      - sac
      - dsac
      - sac_ca
      - sac_dica_s=4_te=8_unet=(256, 256)

    在 run 目录内部，优先选择：
      1) policy.pth
      2) policy_best.pth
      3) 按最大 epoch 选择 policy_epoch{N}.pth

    Args:
        node_num: 节点数量
        quality_ratio: 质量系数 (会按三位小数格式化来匹配目录)
        algo_prefix: 算法目录前缀
        runs_root: runs 根目录名

    Returns:
        str: 选定的 checkpoint 文件的绝对或相对路径

    Raises:
        FileNotFoundError: 未找到匹配的目录或 checkpoint 文件
    """
    ratio_str = f"{quality_ratio:.3f}"
    base_dir = os.path.join(runs_root, f"{int(node_num)}_{ratio_str}")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"未找到目录: {base_dir}")

    # 找到所有匹配的算法目录
    # 区分标准算法和带变体参数的算法
    algo_dirs: List[str] = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full):
            # 对于标准算法，使用精确匹配
            if algo_prefix in ["sac", "ppo", "ddpg", "td3", "dsac", "fifo", "rand", "fixed", "greedy"]:
                if name == algo_prefix:
                    algo_dirs.append(full)
            # 对于带变体参数的算法，使用前缀匹配
            elif algo_prefix in ["sac_ca", "sac_dica"]:
                if name.startswith(algo_prefix):
                    algo_dirs.append(full)
            # 其他情况使用精确匹配
            else:
                if name == algo_prefix:
                    algo_dirs.append(full)

    if not algo_dirs:
        raise FileNotFoundError(f"未在 {base_dir} 下找到算法目录 '{algo_prefix}'")

    # 在所有匹配的算法目录中，收集其下的 run 目录
    run_dirs: List[str] = []
    for ad in algo_dirs:
        for name in os.listdir(ad):
            full = os.path.join(ad, name)
            if os.path.isdir(full):
                run_dirs.append(full)
    if not run_dirs:
        raise FileNotFoundError(f"未在 {algo_dirs} 下找到任何运行目录")

    # 选择最近修改时间的 run 目录
    run_dirs.sort(key=lambda p: os.path.getmtime(p))
    latest_run = run_dirs[-1]

    # 在 run 目录中选择 checkpoint 文件
    candidates = [
        os.path.join(latest_run, "policy.pth"),
        os.path.join(latest_run, "policy_best.pth"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # fallback: 查找最高 epoch
    epoch_files = [f for f in os.listdir(latest_run) if f.startswith("policy_epoch") and f.endswith(".pth")]
    if not epoch_files:
        raise FileNotFoundError(f"在 {latest_run} 未找到任何 checkpoint 文件")

    def _epoch_num(name: str) -> int:
        import re
        m = re.search(r"policy_epoch(\d+)\.pth", name)
        return int(m.group(1)) if m else -1

    epoch_files.sort(key=_epoch_num)
    return os.path.join(latest_run, epoch_files[-1])


def save_experiment_summary(algo: str, node_num: int, lambda_val: float, 
                           quality_ratio: Optional[float], seed: int, 
                           metrics: Dict[str, float], 
                           summary_file: str = "data/summary.jsonl") -> None:
    """
    将实验的summary信息追加到全局summary.jsonl文件中。
    使用文件锁支持并发写入。
    
    Args:
        algo: 算法名称前缀（如"sac", "ppo", "fifo"等）
        node_num: 节点数量
        lambda_val: 任务到达率
        quality_ratio: 质量系数（RL算法有，非RL算法为None）
        seed: 随机种子
        metrics: 关键指标字典，应包含 avg_response_time 和 avg_quality
        summary_file: summary文件路径，默认为"data/summary.jsonl"
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    
    # 构建summary记录
    record = {
        "time": datetime.now().isoformat(),
        "seed": seed,
        "algo": algo,
        "node_num": node_num,
        "lambda": lambda_val,
        "quality_ratio": quality_ratio,
        "metrics": metrics
    }
    
    # 使用文件锁安全写入
    with open(summary_file, 'a') as f:
        # 获取独占锁
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            # 写入一行JSON
            f.write(json.dumps(record) + '\n')
            f.flush()
        finally:
            # 释放锁
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    print(f"实验summary已追加到: {summary_file}")