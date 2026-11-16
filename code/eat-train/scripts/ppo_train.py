from __future__ import annotations

import os
import torch
from tianshou.env import SubprocVectorEnv
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

# 导入 datetime 模块
from datetime import datetime

from env import get_config, EdgeEnv
from trainer.ppo import get_trainer
from config import get_train_config, get_global_config, set_global_config

def make_vec_env(num: int, seed: int, num_nodes: int = 32):
    """Create vectorized environment.
    
    Args:
        num: Number of parallel environments
        seed: Random seed
        num_nodes: Number of nodes in the cluster (16, 32, 48, or 64)
    """
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
    
    # Hyperparameters
    cfg_train = get_train_config()
    seed = cfg_train.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_num = cfg_train.train_num
    test_num = cfg_train.test_num
    lr = 3e-4
    epoch = 1500
    step_per_epoch = 10000
    episode_per_collect = 8
    repeat_per_collect = 8
    batch_size = 512

    # Envs
    train_envs = make_vec_env(train_num, seed=seed, num_nodes=num_nodes)
    test_envs = make_vec_env(test_num, seed=seed + 1000, num_nodes=num_nodes)

    # 计算质量系数
    from env.config import get_config
    env_cfg = get_config(num_nodes=num_nodes)
    reward_q_weight = env_cfg.reward_q_weight
    reward_t1_weight = env_cfg.reward_t1_weight
    reward_t2_weight = env_cfg.reward_t2_weight
    quality_ratio = reward_q_weight / (reward_q_weight + reward_t1_weight + reward_t2_weight)
    
    # Logger - 日志目录包含节点数和质量系数
    log_base_path = os.path.join("runs", f"{num_nodes}"+"_"+f"{quality_ratio:.3f}", "ppo")
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_base_path, run_name)
    # 3. SummaryWriter 会自动创建目录，所以 os.makedirs 不是必需的
    writer = SummaryWriter(log_dir)
    logger = TensorboardLogger(writer)

    # Trainer
    trainer, policy = get_trainer(
        train_envs=train_envs,
        test_envs=test_envs,
        device=device,
        lr=lr,
        epoch=epoch,
        step_per_epoch=step_per_epoch,
        repeat_per_collect=repeat_per_collect,
        episode_per_collect=episode_per_collect,
        batch_size=batch_size,
        test_num=test_num,
        logger=logger,
        save_best_fn=lambda p: torch.save(p.state_dict(), os.path.join(log_dir, "policy_best.pth")),
        save_checkpoint_fn=lambda epoch, env_step, gradient_step: (
            torch.save(policy.state_dict(), os.path.join(log_dir, f"policy_epoch{epoch}.pth"))
            if (epoch % max(1, cfg_train.save_interval_epochs) == 0) else None
        ),
        actor_hidden_sizes=cfg_train.actor_hidden_sizes,
        critic_hidden_sizes=cfg_train.critic_hidden_sizes,
    )

    # 加载和保存检查点的路径也应与日志目录关联
    ckpt_path = os.path.join(log_dir, "policy.pth")
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)
        policy.load_state_dict(state_dict)
        print(f"Loaded checkpoint from: {ckpt_path}")

    # Train
    result = trainer.run()
    print("Training finished:", result)

    # Save checkpoint
    torch.save(policy.state_dict(), ckpt_path)
    print(f"Saved checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
