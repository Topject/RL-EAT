"""
贪心策略评估脚本

策略：
1. exec_flag 固定为 1.0（总是执行）
2. step 和 task 维度按指定步长遍历所有可能的组合
3. 使用 env.preview_reward() 计算每个候选action的预期reward
4. 选择reward最大的action执行
"""

from __future__ import annotations

import os
import csv
import numpy as np
from typing import Optional, Tuple
from datetime import datetime

from env import EdgeEnv, get_config
from utils import save_eval_details, save_eval_summary, save_experiment_summary


class GreedyPolicy:
    """贪心策略：遍历所有可能的action，选择reward最大的"""
    
    def __init__(self, step_grid: int = 10, task_grid: int = 8):
        """初始化贪心策略
        
        Args:
            step_grid: step维度的网格数（将[-1, 1]分成多少份）
            task_grid: task维度的网格数（将[-1, 1]分成多少份）
        """
        self.step_grid = step_grid
        self.task_grid = task_grid
        
        # 预计算网格点
        self.step_values = np.linspace(-1.0, 1.0, step_grid)
        self.task_values = np.linspace(-1.0, 1.0, task_grid)
        
        print(f"贪心策略配置:")
        print(f"  - step网格数: {step_grid}, 范围: [{self.step_values[0]:.2f}, {self.step_values[-1]:.2f}]")
        print(f"  - task网格数: {task_grid}, 范围: [{self.task_values[0]:.2f}, {self.task_values[-1]:.2f}]")
        print(f"  - 总候选数: {step_grid * task_grid}")
    
    def select_action(self, env: EdgeEnv) -> np.ndarray:
        """选择最优action
        
        Args:
            env: 环境实例
            
        Returns:
            最优的3维action [exec_flag, step, task]
        """
        # 如果队列为空，跳过
        if len(env.queue) == 0:
            return np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        
        best_action = None
        best_reward = -float('inf')
        
        # 遍历所有候选action
        for step_val in self.step_values:
            for task_val in self.task_values:
                # 构造候选action（exec_flag固定为1.0）
                candidate_action = np.array([1.0, step_val, task_val], dtype=np.float32)
                
                # 预览reward
                reward = env.preview_reward(candidate_action)
                
                # 更新最优
                if reward > best_reward:
                    best_reward = reward
                    best_action = candidate_action
        
        # 如果没有找到合法的action（所有reward都是0），跳过
        if best_reward <= 0.0:
            return np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        
        return best_action


"""参数设置：与其他非RL脚本保持一致（变量配置）"""
MAX_STEPS: int = 0
REPEAT_TIMES: int = 8
RENDER: bool = False
SEED: int = 42
NUM_NODES: int = 32
ALGO_NAME: str = "greedy"
# 任务到达率（lambda），None表示使用环境默认值（根据节点数自动设置）
LAMBDA: float = None
STEP_GRID: int = 10
TASK_GRID: int = 8


def evaluate_greedy_once(
    steps: int,
    render: bool,
    seed: int,
    run_id: int,
    eval_dir: str,
) -> dict:
    """评估一次贪心策略，返回标准结果结构并写入CSV到 eval_dir"""
    # 配置环境
    cfg = get_config()
    cfg.num_nodes = NUM_NODES
    cfg.concat_state = True
    # 如果指定了lambda，手动设置任务到达率
    if LAMBDA is not None:
        cfg.poisson_lambda_per_sec = LAMBDA

    # 策略
    policy = GreedyPolicy(step_grid=STEP_GRID, task_grid=TASK_GRID)

    # CSV 文件路径
    csv_path = os.path.join(eval_dir, f"eval_run_seed{seed}_id{run_id}.csv")
    tasks_csv_path = os.path.join(eval_dir, f"eval_tasks_seed{seed}_id{run_id}.csv")

    # 环境
    env = EdgeEnv(
        config=cfg,
        seed=seed,
        enable_eval=True,
        eval_csv_path=csv_path,
        eval_tasks_csv_path=tasks_csv_path,
        enable_render=bool(render),
    )

    obs, info = env.reset(seed=seed)
    done = False
    truncated = False
    total_reward = 0.0
    total_quality = 0.0
    quality_count = 0
    t = 0

    while not (done or truncated):
        action = policy.select_action(env)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
        # Accumulate quality if present
        if "quality" in info and info["quality"] is not None:
            total_quality += float(info["quality"])
            quality_count += 1
        t += 1
        if steps > 0 and t >= steps:
            break

    completed_count, avg_latency = env.get_completed_stats()
    avg_quality = total_quality / max(1, quality_count) if quality_count > 0 else 0.0
    return {
        'total_reward': float(total_reward),
        'completed_count': int(completed_count),
        'avg_resp': float(avg_latency),
        'avg_quality': avg_quality,
        'steps': int(t),
    }


def main():
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join("data", f"{NUM_NODES}", ALGO_NAME, run_name)
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    for i in range(REPEAT_TIMES):
        cur_seed = int(SEED) + i
        print(f"\n==== Run {i+1}/{REPEAT_TIMES} with seed={cur_seed} ====")
        result = evaluate_greedy_once(
            steps=MAX_STEPS,
            render=bool(RENDER),
            seed=cur_seed,
            run_id=i,
            eval_dir=out_dir,
        )
        result['seed'] = cur_seed
        result['run_id'] = i
        all_results.append(result)

    avg_reward = float(np.mean([r['total_reward'] for r in all_results])) if all_results else 0.0
    avg_completed = float(np.mean([r['completed_count'] for r in all_results])) if all_results else 0.0
    avg_resp_time = float(np.mean([r['avg_resp'] for r in all_results])) if all_results else 0.0
    avg_quality = float(np.mean([r['avg_quality'] for r in all_results])) if all_results else 0.0
    avg_steps = float(np.mean([r['steps'] for r in all_results])) if all_results else 0.0

    save_eval_details(out_dir, all_results)
    save_eval_summary(out_dir, REPEAT_TIMES, avg_reward, avg_completed, avg_resp_time, avg_quality, avg_steps)
    
    # 保存到全局summary.jsonl
    # 获取实际使用的lambda值
    lambda_val = LAMBDA if LAMBDA is not None else 0.3  # 默认值，与__post_init__保持一致
    key_metrics = {
        "avg_response_time": avg_resp_time,
        "avg_quality": avg_quality
    }
    save_experiment_summary(
        algo=ALGO_NAME,
        node_num=NUM_NODES,
        lambda_val=lambda_val,
        quality_ratio=None,  # 非RL算法没有质量比率
        seed=SEED,
        metrics=key_metrics
    )


if __name__ == "__main__":
    main()
