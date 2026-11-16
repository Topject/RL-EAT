from __future__ import annotations

from typing import Any, List

import numpy as np


def compute_and_log_epoch_metrics(trainer: Any, writer: Any, cur_epoch: int) -> None:
    """
    记录训练自定义指标到 TensorBoard：
    - 每轮次完成任务总数量(train/completed_tasks_per_epoch)
    - 每轮次任务平均响应时间(train/avg_response_time)
    - 连续三维动作的均值与标准差(train/action{d}_mean, train/action{d}_std)
    - 每轮次 q_min 的平均值(train/avg_min_q)
    - 每轮次 ddim_eta 的值(train/ddim_eta)

    说明：
    - 从 trainer.train_collector.buffer 取样本，解析 Batch.act 与 Batch.info。
    - info 中读取 completed_count 与 completed_latencies（在 env 中由 step 填充）。
    """
    try:
        buf = getattr(trainer, "train_collector").buffer
        buf_len = int(len(buf))
        if buf_len <= 0:
            return

        batch = buf.sample(buf_len)

        # 从 trainer.policy.actor_optim 的参数组中获取当前学习率
        if hasattr(trainer.policy, "actor_optim"):
            actor_lr = trainer.policy.actor_optim.param_groups[0]['lr']
            writer.add_scalar("train/lr_actor", actor_lr, cur_epoch)

        if hasattr(trainer.policy, "critic1_optim"):
            critic_lr = trainer.policy.critic1_optim.param_groups[0]['lr']
            writer.add_scalar("train/lr_critic", critic_lr, cur_epoch)

        # 动作统计（取前三维）
        if hasattr(batch[0], "act") and batch[0].act is not None:
            act_np = batch[0].act
            try:
                act_np = act_np.asnumpy()  # 兼容 tianshou Batch/torch 张量
            except Exception:
                pass
            act_np = np.asarray(act_np)
            if act_np.ndim >= 2 and act_np.shape[1] >= 1:
                dims = min(3, int(act_np.shape[1]))
                for d in range(dims):
                    writer.add_scalar(f"train/action{d}_mean", float(np.nanmean(act_np[:, d])), cur_epoch)
                    writer.add_scalar(f"train/action{d}_std", float(np.nanstd(act_np[:, d])), cur_epoch)

        # 完成任务数量与响应时间（聚合）
        total_completed = 0
        total_latency_sum = 0.0
        total_quality_sum = 0.0
        quality_count = 0
        info_batch = getattr(batch[0], "info", None)
        if info_batch is not None:
            # completed_count
            try:
                completed_counts = info_batch.get("completed_count", None)
            except Exception:
                completed_counts = getattr(info_batch, "completed_count", None)
            if completed_counts is not None:
                try:
                    total_completed = int(np.nansum(np.asarray(completed_counts)))
                except Exception:
                    try:
                        total_completed = int(sum(int(x or 0) for x in completed_counts))
                    except Exception:
                        pass
            # completed_latency_sum
            latency_sums = info_batch.get("completed_latency_sum", None)
            if latency_sums is not None:
                try:
                    total_latency_sum = float(np.nansum(np.asarray(latency_sums)))
                except Exception:
                    try:
                        total_latency_sum = float(sum(float(x or 0.0) for x in latency_sums))
                    except Exception:
                        pass
            # quality
            quality_values = info_batch.get("quality", None)
            if quality_values is not None:
                try:
                    quality_array = np.asarray(quality_values)
                    valid_mask = ~np.isnan(quality_array)
                    total_quality_sum = float(np.sum(quality_array[valid_mask]))
                    quality_count = int(np.sum(valid_mask))
                except Exception:
                    try:
                        for q in quality_values:
                            if q is not None and not np.isnan(q):
                                total_quality_sum += float(q)
                                quality_count += 1
                    except Exception:
                        pass

        avg_latency = float(total_latency_sum / max(1, total_completed)) if total_completed > 0 else 0.0
        avg_quality = float(total_quality_sum / max(1, quality_count)) if quality_count > 0 else 0.0
        writer.add_scalar("train/completed_tasks_per_epoch", total_completed, cur_epoch)
        writer.add_scalar("train/avg_response_time", avg_latency, cur_epoch)
        writer.add_scalar("train/avg_quality", avg_quality, cur_epoch)

        # 仅在 SACDDIMPolicy 中记录 q_min 的 epoch 平均值
        try:
            policy = getattr(trainer, "policy", None)
            if policy is not None and hasattr(policy, "pop_avg_min_q"):
                avg_min_q = policy.pop_avg_min_q()
                if avg_min_q is not None:
                    writer.add_scalar("train/avg_min_q", float(avg_min_q), cur_epoch)
        except Exception:
            pass

        # 仅在 DDIM 变体中记录 eta（如果 actor 提供 eta_param）
        try:
            policy = getattr(trainer, "policy", None)
            actor = getattr(policy, "actor", None) if policy is not None else None
            eta_param = getattr(actor, "eta_param", None)
            if eta_param is not None:
                try:
                    # eta = sigmoid(eta_param)
                    eta_raw = float(eta_param.detach().float().cpu().item())
                    eta = 1.0 / (1.0 + np.exp(-eta_raw))
                    writer.add_scalar("train/ddim_eta", float(eta), cur_epoch)
                except Exception:
                    pass
        except Exception:
            pass
        
        try:
            policy = getattr(trainer, "policy", None)
            if policy is not None and hasattr(policy, "pop_metrics"):
                # 1. 从 policy 中"弹出"所有缓存的梯度数据
                gradient_data_points = policy.pop_metrics()

                # 2. 遍历每个时间点的数据
                for data_point in gradient_data_points:
                    if "step" not in data_point:
                        continue
                    
                    # 获取该数据点对应的全局更新步数作为 X 轴
                    step = data_point.pop("step")
                    
                    # 3. 将该时间点的所有梯度指标写入 TensorBoard
                    for metric_name, value in data_point.items():
                        # metric_name 已经是 'critic1_total_norm' 这样的完整标签
                        writer.add_scalar(metric_name, value, step)

        except Exception as e:
            print(f"[warn] failed to log gradient metrics: {e}")
    except Exception as e:
        print(f"[warn] failed to log custom epoch metrics: {e}")
