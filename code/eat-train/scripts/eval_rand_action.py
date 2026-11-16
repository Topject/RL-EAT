from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from tianshou.data import Batch
from tianshou.env import DummyVectorEnv

from env import EdgeEnv, get_config
from config import get_train_config
from utils import save_eval_details, save_eval_summary, save_experiment_summary
from datetime import datetime


# ===== 用户可编辑参数 =====
# 最大步数（0 表示直到环境 done/truncated）
MAX_STEPS: int = 0
# 重复评估的次数（不同种子）。<=1 表示只评估一次。
REPEAT_TIMES: int = 8
# 是否渲染（render 仅在 eval=True 时启用）
RENDER: bool = False
# 随机种子
SEED: int = 42
# 节点数量（16/32/48/64）
NUM_NODES: int = 32
ALGO_NAME: str = "rand"
# 任务到达率（lambda），None表示使用环境默认值（根据节点数自动设置）
LAMBDA: float = None

def evaluate_rand(steps: int, render: bool, seed: Optional[int], run_id: int = 0, eval_dir: str = "") -> dict:

    # Build single env for evaluation with CSV + optional render
    cfg = get_config()
    cfg.concat_state = True
    cfg.num_nodes = NUM_NODES
    # 如果指定了lambda，手动设置任务到达率
    if LAMBDA is not None:
        cfg.poisson_lambda_per_sec = LAMBDA
    out_dir = eval_dir or os.path.join("data", f"{NUM_NODES}", ALGO_NAME)
    os.makedirs(out_dir, exist_ok=True)
    csv_file = os.path.join(out_dir, f"eval_run_seed{seed}_id{run_id}.csv")
    env = EdgeEnv(
        config=cfg,
        seed=seed,
        enable_eval=True,
        enable_render=bool(render),
        eval_csv_path=csv_file,
        eval_tasks_csv_path=os.path.join(out_dir, f"eval_tasks_seed{seed}_id{run_id}.csv"),
    )

    obs, info = env.reset(seed=seed)
    done = False
    trunc = False
    t = 0
    total_reward = 0.0
    total_quality = 0.0
    quality_count = 0

    while not (done or trunc):
        # Prepare obs batch for policy
        if isinstance(obs, tuple):
            # Expect concatenated; but in case concat_state=False, handle tuple -> flatten
            nodes, queue = obs
            obs_vec = np.concatenate([nodes.flatten(), queue.flatten()]).astype(np.float32)
        else:
            obs_vec = np.asarray(obs, dtype=np.float32)
        batch = Batch(obs=obs_vec[None, :])

        # 随机动作 每个维度从[-1,1]随机采样
        act = np.random.uniform(low=-1.0, high=1.0, size=3)
        act = np.asarray(act, dtype=np.float32)

        obs, rew, done, trunc, info = env.step(act)
        total_reward += float(rew)
        # Accumulate quality if present
        if "quality" in info and info["quality"] is not None:
            total_quality += float(info["quality"])
            quality_count += 1
        t += 1
        if steps > 0 and t >= steps:
            break

    # 输出本次评估的任务完成统计
    completed_count, avg_resp = env.get_completed_stats()
    avg_quality = total_quality / max(1, quality_count) if quality_count > 0 else 0.0
    print(f"Evaluation finished after {t} steps. "
          f"total_reward={total_reward:.4f}, completed={completed_count}, avg_resp={avg_resp:.4f}s, avg_quality={avg_quality:.4f}")
    print(f"Step-wise CSV saved to: {csv_file}")

    return {
        'total_reward': total_reward,
        'completed_count': completed_count,
        'avg_resp': avg_resp,
        'avg_quality': avg_quality,
        'steps': t
    }

def main():

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join("data", f"{NUM_NODES}", ALGO_NAME, run_name)
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    # 多次评估，种子递增
    for i in range(REPEAT_TIMES):
        cur_seed = int(SEED) + i
        print(f"\n==== Run {i+1}/{REPEAT_TIMES} with seed={cur_seed} ====")
        result = evaluate_rand(steps=MAX_STEPS, render=bool(RENDER), seed=cur_seed, run_id=i, eval_dir=out_dir)
        result['seed'] = cur_seed
        result['run_id'] = i
        all_results.append(result)

    avg_reward = float(np.mean([r['total_reward'] for r in all_results])) if all_results else 0.0
    avg_completed = float(np.mean([r['completed_count'] for r in all_results])) if all_results else 0.0
    avg_resp_time = float(np.mean([r['avg_resp'] for r in all_results])) if all_results else 0.0
    avg_quality = float(np.mean([r['avg_quality'] for r in all_results])) if all_results else 0.0
    avg_steps = float(np.mean([r['steps'] for r in all_results])) if all_results else 0.0

    print(f"Average Completed count: {avg_completed}")
    print(f"Average response time: {avg_resp_time}")
    print(f"Average total reward: {avg_reward}")

    # 保存评估结果到 data/{NUM_NODES}/{ALGO_NAME}
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




