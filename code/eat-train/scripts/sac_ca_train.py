from __future__ import annotations

import os
import torch
from tianshou.env import SubprocVectorEnv
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from trainer.metrics import compute_and_log_epoch_metrics
from datetime import datetime

from env import get_config, EdgeEnv
from config import get_train_config, get_global_config, set_global_config
from trainer.casac.trainer import get_trainer


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
    set_global_config(node_num = 64, max_visible_tasks=8)
    cfg_global = get_global_config()
    num_nodes = cfg_global.node_num
    
    cfg_train = get_train_config()
    seed = cfg_train.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_num = cfg_train.train_num
    test_num = cfg_train.test_num
    lr = 3e-4
    min_lr = 3e-6
    epoch = 1500
    step_per_epoch = 1000
    batch_size = 4096
    step_per_collect = 200
    update_per_step = 2/cfg_train.train_num
    
    train_envs = make_vec_env(train_num, seed=seed, num_nodes=num_nodes)
    test_envs = make_vec_env(test_num, seed=seed + 1000, num_nodes=num_nodes)

    # 计算质量系数
    env_cfg = get_config(num_nodes=num_nodes)
    reward_q_weight = env_cfg.reward_q_weight
    reward_t1_weight = env_cfg.reward_t1_weight
    reward_t2_weight = env_cfg.reward_t2_weight
    quality_ratio = reward_q_weight / (reward_q_weight + reward_t1_weight + reward_t2_weight)

    log_base_path = os.path.join("runs", f"{num_nodes}"+"_"+f"{quality_ratio:.3f}", "sac_ca")
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_base_path, run_name)
    writer = SummaryWriter(log_dir)
    logger = TensorboardLogger(writer)
    print(f"Logs will be saved to: {log_dir}")

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
        save_checkpoint_fn=None,
        #embed_dim=cfg_train.state_embed_dim,
        actor_hidden_sizes=cfg_train.actor_hidden_sizes,
        critic_hidden_sizes=cfg_train.critic_hidden_sizes,
    )

    # 设置在每个 epoch 结束时调用的保存/记录函数
    def _on_epoch_end(cur_epoch: int, env_step: int, gradient_step: int):
        compute_and_log_epoch_metrics(trainer, writer, cur_epoch)
        if cur_epoch % max(1, cfg_train.save_interval_epochs) == 0:
            torch.save(policy.state_dict(), os.path.join(log_dir, f"policy_epoch{cur_epoch}.pth"))

    # 将回调绑定到 trainer（OffpolicyTrainer 接受 save_checkpoint_fn 回调）
    trainer.save_checkpoint_fn = _on_epoch_end

    ckpt_path = os.path.join(log_dir, "policy.pth")
    if os.path.exists(ckpt_path):
        try:
            state_dict = torch.load(ckpt_path, map_location=device)
            policy.load_state_dict(state_dict)
            print(f"Successfully loaded checkpoint from: {ckpt_path}")
        except Exception as e:
            print(f"Could not load checkpoint from {ckpt_path}: {e}")

    print("Starting training (Cross-Attention SAC)...")
    result = trainer.run()
    print("Training finished:", result)

    torch.save(policy.state_dict(), ckpt_path)
    print(f"Saved final policy checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()


