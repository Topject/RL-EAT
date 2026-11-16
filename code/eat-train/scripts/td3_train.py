from __future__ import annotations

import os
import torch
from tianshou.env import SubprocVectorEnv
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

# 导入 datetime 模块
from datetime import datetime

# 假设您的环境代码在名为 env.py 的文件中
from env import get_config, EdgeEnv
from config import get_train_config, get_global_config, set_global_config
# 保持您原有的 trainer 导入方式
from trainer.td3.trainer import get_trainer

def make_vec_env(num: int, seed: int, num_nodes: int = 32):
    cfg = get_config(num_nodes=num_nodes)
    cfg.concat_state = True

    env_fns = []
    for i in range(num):
        current_seed = seed + i
        env_fn = lambda s=current_seed: EdgeEnv(config=cfg, seed=s)
        env_fns.append(env_fn)

    return SubprocVectorEnv(env_fns)


def main():
    from utils import set_best_gpu
    set_best_gpu()

    # ===== 实验配置：在这里设置节点数量和可见任务数 =====
    set_global_config(node_num=64, max_visible_tasks=8)
    cfg_global = get_global_config()
    num_nodes = cfg_global.node_num

    # --- 超参数定义 ---
    cfg_train = get_train_config()
    seed = cfg_train.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_num = cfg_train.train_num  # 并行训练环境的数量
    test_num = cfg_train.test_num   # 并行测试环境的数量
    lr = 3e-4
    epoch = 1500
    step_per_epoch = 1000  # 每个 epoch 固定 1000 step
    batch_size = 512
    # step-based: 每次收集 1000 step，按比例更新 4/1000
    step_per_collect = 1000
    update_per_step = 1

    # --- 环境初始化 ---

    train_envs = make_vec_env(train_num, seed=seed, num_nodes=num_nodes)
    test_envs = make_vec_env(test_num, seed=seed + 1000, num_nodes=num_nodes)

    # 计算质量系数
    env_cfg = get_config(num_nodes=num_nodes)
    reward_q_weight = env_cfg.reward_q_weight
    reward_t1_weight = env_cfg.reward_t1_weight
    reward_t2_weight = env_cfg.reward_t2_weight
    quality_ratio = reward_q_weight / (reward_q_weight + reward_t1_weight + reward_t2_weight)

    # --- 日志记录器 (Logger) 设置 ---
    # 1. 定义一个基础路径 - 日志目录包含节点数和质量系数
    log_base_path = os.path.join("runs", f"{num_nodes}"+"_"+f"{quality_ratio:.3f}", "td3")
    # 2. 创建一个基于当前时间的唯一目录名
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_base_path, run_name)
    # 3. SummaryWriter 会自动创建目录
    writer = SummaryWriter(log_dir)
    logger = TensorboardLogger(writer)
    print(f"Logs will be saved to: {log_dir}")

    # --- 获取训练器实例 ---
    trainer, policy = get_trainer(
        train_envs=train_envs,
        test_envs=test_envs,
        device=device,
        lr=lr,
        epoch=epoch,
        step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect,
        update_per_step=update_per_step,
        batch_size=batch_size,
        test_num=test_num,
        logger=logger,
        save_best_fn=lambda p: torch.save(p.state_dict(), os.path.join(log_dir, "policy_best.pth")),
        save_checkpoint_fn=lambda epoch, env_step, gradient_step: (
            torch.save(policy.state_dict(), os.path.join(log_dir, f"policy_epoch{epoch}.pth"))
            if (epoch % max(1, cfg_train.save_interval_epochs) == 0) else None
        ),
        hidden_sizes=cfg_train.actor_hidden_sizes,
    )

    # --- 检查点加载 ---
    # 加载和保存检查点的路径也应与日志目录关联
    ckpt_path = os.path.join(log_dir, "policy.pth")
    if os.path.exists(ckpt_path):
        try:
            state_dict = torch.load(ckpt_path, map_location=device)
            policy.load_state_dict(state_dict)
            print(f"Successfully loaded checkpoint from: {ckpt_path}")
        except Exception as e:
            print(f"Could not load checkpoint from {ckpt_path}: {e}")

    # --- 开始训练 ---
    print("Starting training...")
    result = trainer.run()
    print("Training finished:", result)

    # --- 保存最终模型 ---
    torch.save(policy.state_dict(), ckpt_path)
    print(f"Saved final policy checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
