from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from tianshou.data import Batch
from tianshou.env import DummyVectorEnv

from env import EdgeEnv, get_config
from config import get_train_config
from utils import set_best_gpu, save_eval_details, save_eval_summary, find_latest_ckpt, save_experiment_summary


# ===== 用户可编辑参数（仅 dca sac）=====
# 设置 checkpoint 文件或目录。留空则自动根据节点数与质量系数查找最新的。
CKPT_PATH: str = ""
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
# 质量系数（用于在 runs 中自动匹配目录）
QUALITY_RATIO: float = 0.600
# 任务到达率（lambda），None表示使用环境默认值（根据节点数自动设置）
LAMBDA: float = None


def _find_checkpoint(ckpt_path: str) -> str:
    if os.path.isdir(ckpt_path):
        # Prefer policy.pth, then policy_best.pth, then latest policy_epoch*.pth
        candidates = [
            os.path.join(ckpt_path, "policy.pth"),
            os.path.join(ckpt_path, "policy_best.pth"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        # fallback: latest epoch
        epoch_files = [f for f in os.listdir(ckpt_path) if f.startswith("policy_epoch") and f.endswith(".pth")]
        if not epoch_files:
            raise FileNotFoundError(f"No checkpoint found in directory: {ckpt_path}")
        # sort by epoch number
        def _epoch_num(name: str) -> int:
            try:
                import re
                m = re.search(r"policy_epoch(\d+)\.pth", name)
                return int(m.group(1)) if m else -1
            except Exception:
                return -1
        epoch_files.sort(key=_epoch_num)
        return os.path.join(ckpt_path, epoch_files[-1])
    else:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path


def _get_eval_dir_from_ckpt(ckpt_file: str) -> str:
    """将以 runs 开头的 ckpt 目录映射为以 data 开头的评估输出目录，并确保存在。"""
    ckpt_dir = os.path.dirname(ckpt_file)
    parts = ckpt_dir.split(os.sep)
    if "runs" in parts:
        parts[parts.index("runs")] = "data"
        out_dir = os.sep.join(parts)
    else:
        out_dir = os.path.join("data", ckpt_dir)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _build_policy_dcasac(device: str):
    # Recreate policy with the same architecture as training via dcasac trainer.get_trainer
    cfg_env = get_config(num_nodes=NUM_NODES)
    cfg_env.concat_state = True

    def _env_factory():
        return EdgeEnv(config=cfg_env, seed=None)

    envs = DummyVectorEnv([_env_factory])

    from trainer.dicasac.trainer import get_trainer
    cfg_train = get_train_config()
    trainer, policy = get_trainer(
        train_envs=envs,
        test_envs=None,
        device=device,
        lr=3e-4,
        epoch=1,
        step_per_epoch=1,
        step_per_collect=1,
        update_per_step=0.0,
        batch_size=64,
        test_num=0,
        logger=None,
        save_best_fn=None,
        save_checkpoint_fn=None,
        actor_hidden_sizes=cfg_train.actor_hidden_sizes,
        critic_hidden_sizes=cfg_train.critic_hidden_sizes,
        unet_hidden_sizes=cfg_train.unet_hidden_size,
        state_embed_dim=cfg_train.state_embed_dim,
        diffusion_timesteps=cfg_train.diffusion_steps,
        time_embed_dim=cfg_train.diffusion_time_embed_dim,
    )
    
    return policy


def evaluate_dcasac(ckpt: str, steps: int, render: bool, seed: Optional[int], run_id: int = 0) -> dict:
    """
    评估一次dcasac策略，返回评估结果字典。
    
    Returns:
        dict: 包含 'total_reward', 'completed_count', 'avg_resp', 'steps' 的字典
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_file = _find_checkpoint(ckpt)
    ckpt_dir = os.path.dirname(ckpt_file)
    eval_dir = _get_eval_dir_from_ckpt(ckpt_file)
    print(f"Using checkpoint: {ckpt_file}")

    # Build policy and load weights (dca sac)
    policy = _build_policy_dcasac(device=device)
    state = torch.load(ckpt_file, map_location=device)
    policy.load_state_dict(state)
    policy.eval()

    # Build single env for evaluation with CSV + optional render
    cfg = get_config(num_nodes=NUM_NODES)
    cfg.concat_state = True
    # 如果指定了lambda，手动设置任务到达率
    if LAMBDA is not None:
        cfg.poisson_lambda_per_sec = LAMBDA
    # 每次评估使用不同的文件名，避免覆盖（保存到 data 开头的目录）
    csv_path = os.path.join(eval_dir, f"eval_run_seed{seed}_id{run_id}.csv")
    tasks_csv_path = os.path.join(eval_dir, f"eval_tasks_seed{seed}_id{run_id}.csv")
    env = EdgeEnv(
        config=cfg,
        seed=seed,
        enable_eval=True,
        enable_render=bool(render),
        eval_csv_path=csv_path,
        eval_tasks_csv_path=tasks_csv_path,
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
        with torch.no_grad():
            out = policy.forward(batch)
        act = out.act
        if hasattr(act, "cpu"):
            act = act.cpu().numpy()
        act = np.asarray(act, dtype=np.float32)[0]

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
    print(f"Step-wise CSV saved to: {csv_path}")
    
    return {
        'total_reward': total_reward,
        'completed_count': completed_count,
        'avg_resp': avg_resp,
        'avg_quality': avg_quality,
        'steps': t
    }


def main():
    # 自动选择最佳GPU
    set_best_gpu()

    from config import set_global_config
    set_global_config(node_num = NUM_NODES, max_visible_tasks=8)

    if CKPT_PATH:
        ckpt_file = _find_checkpoint(CKPT_PATH)
    else:
        ckpt_file = find_latest_ckpt(node_num=NUM_NODES, quality_ratio=QUALITY_RATIO, algo_prefix="sac_dica", runs_root="runs")
    ckpt_dir = os.path.dirname(ckpt_file)
    eval_dir = _get_eval_dir_from_ckpt(ckpt_file)
    
    if REPEAT_TIMES <= 1:
        # 单次评估
        evaluate_dcasac(ckpt=ckpt_file, steps=MAX_STEPS, render=bool(RENDER), seed=SEED, run_id=0)
        return
    
    # 多次评估，种子递增
    all_results = []
    for i in range(REPEAT_TIMES):
        cur_seed = int(SEED) + i
        print(f"\n==== Run {i+1}/{REPEAT_TIMES} with seed={cur_seed} ====")
        result = evaluate_dcasac(ckpt=ckpt_file, steps=MAX_STEPS, render=bool(RENDER), seed=cur_seed, run_id=i)
        result['seed'] = cur_seed
        result['run_id'] = i
        all_results.append(result)
    
    # 计算并输出平均统计
    avg_reward = np.mean([r['total_reward'] for r in all_results])
    avg_completed = np.mean([r['completed_count'] for r in all_results])
    avg_resp_time = np.mean([r['avg_resp'] for r in all_results])
    avg_quality = np.mean([r['avg_quality'] for r in all_results])
    avg_steps = np.mean([r['steps'] for r in all_results])
    
    print("\n" + "="*60)
    print(f"重复测试 {REPEAT_TIMES} 次的平均结果:")
    print(f"  平均总奖励:     {avg_reward:.4f}")
    print(f"  平均完成数量:   {avg_completed:.2f}")
    print(f"  平均响应时间:   {avg_resp_time:.4f}s")
    print(f"  平均质量:       {avg_quality:.4f}")
    print(f"  平均步数:       {avg_steps:.2f}")
    print("="*60)
    
    # 保存评估结果到CSV
    save_eval_details(eval_dir, all_results)
    save_eval_summary(eval_dir, REPEAT_TIMES, avg_reward, avg_completed, avg_resp_time, avg_quality, avg_steps)
    
    # 保存到全局summary.jsonl
    # 获取实际使用的lambda值
    lambda_val = LAMBDA if LAMBDA is not None else 0.3  # 默认值，与__post_init__保持一致
    key_metrics = {
        "avg_response_time": avg_resp_time,
        "avg_quality": avg_quality
    }
    save_experiment_summary(
        algo="sac_dica",
        node_num=NUM_NODES,
        lambda_val=lambda_val,
        quality_ratio=QUALITY_RATIO,
        seed=SEED,
        metrics=key_metrics
    )


if __name__ == "__main__":
    main()




