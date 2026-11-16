from __future__ import annotations

import os
from typing import Optional

import numpy as np
from tianshou.data import Batch

from env import EdgeEnv, get_config
from config import set_global_config, set_runtime_config

# ===== 用户可编辑参数 =====
MAX_STEPS: int = 0
REPEAT_TIMES: int = 2
RENDER: bool = False
SEED: int = 42
NUM_NODES: int = 8

csv_path = "runs/real_random"

def evaluate_random(csv_path: str, steps: int, render: bool, seed: Optional[int]) -> tuple:
    # 设置真实环境配置
    set_global_config(node_num=NUM_NODES, max_visible_tasks=8)
    set_runtime_config(
        mode="real",
        node_ips=[
            "172.16.0.2:8000",
            "172.16.0.4:8000",
            "172.16.0.7:8000",
            "172.16.0.8:8000",
            "172.16.0.12:8000",
            "172.16.0.13:8000",
            "172.16.0.14:8000",
            "172.16.0.16:8000"
        ],
        master_addr="127.0.0.1",
        master_port=29500,
        output_dir="outputs_real",
        height=1024,
        width=1024,
        num_inference_steps=50,
        api_timeout=60.0,
        debug=True,  # 启用调试信息
    )
    
    cfg = get_config(num_nodes=NUM_NODES)
    cfg.concat_state = True
    csv_dir = os.path.join(csv_path, f"{NUM_NODES}")
    os.makedirs(csv_dir, exist_ok=True)
    csv_file = os.path.join(csv_dir, "eval_run.csv")
    
    env = EdgeEnv(
        config=cfg,
        seed=seed,
        enable_eval=True,
        enable_render=bool(render),
        eval_csv_path=csv_file,
        eval_tasks_csv_path=os.path.join(csv_dir, "eval_tasks.csv"),
    )

    obs, info = env.reset(seed=seed)
    done = False
    trunc = False
    t = 0
    r = 0
    
    while not (done or trunc):
        if isinstance(obs, tuple):
            nodes, queue = obs
            obs_vec = np.concatenate([nodes.flatten(), queue.flatten()]).astype(np.float32)
        else:
            obs_vec = np.asarray(obs, dtype=np.float32)
        
        # 随机动作
        act = np.random.uniform(low=-1.0, high=1.0, size=3)
        act = np.asarray(act, dtype=np.float32)

        obs, rew, done, trunc, info = env.step(act)
        t += 1
        if steps > 0 and t >= steps:
            break

        r += rew
    
    completed_count, avg_resp = env.get_completed_stats()
    print(f"Evaluation finished after {t} steps. completed={completed_count}, avg_resp={avg_resp:.4f}s")
    print(f"Total reward: {r}") 
    return completed_count, avg_resp, r

def main():
    completed_count_list = []
    avg_resp_list = []
    r_list = []

    for i in range(REPEAT_TIMES):
        cur_seed = int(SEED) + i
        print(f"\n==== Run {i+1}/{REPEAT_TIMES} with seed={cur_seed} ====")
        completed_count, avg_resp, r = evaluate_random(csv_path=csv_path, steps=MAX_STEPS, render=bool(RENDER), seed=cur_seed)
        completed_count_list.append(completed_count)
        avg_resp_list.append(avg_resp)
        r_list.append(r)
    
    print(f"\nAverage Completed count: {np.mean(completed_count_list)}")
    print(f"Average response time: {np.mean(avg_resp_list)}")
    print(f"Average total reward: {np.mean(r_list)}")

if __name__ == "__main__":
    main()

