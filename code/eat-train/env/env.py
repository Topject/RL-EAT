from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import os
import csv

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - optional dependency
    gym = None
    spaces = None  # type: ignore

from .config import EnvConfig, get_config
from .node import Node
from .task import Queue, UserTask, SDTask
from .taskmanager import TaskGenerator, TaskManager
from .scheduler import NodeScheduler
from .node_manager import NodeManager
from config import get_runtime_config


class EdgeEnv(gym.Env):
    """Edge computing environment for SD scheduling.

    Observation: Tuple(nodes_state, queue_state)
      - nodes_state: (num_nodes, 4) -> [remaining_time, model_id, avg_bw, std_bw]
      - queue_state: (l, 3) -> first element is parallelism; rest zeros

    Action: continuous vector of shape (2 + MAX_ACTION_STEP + l)
      - [execute_score, skip_score, steps_scores..., task_scores...]
      - if skip_score > execute_score => skip
      - else choose steps by argmax over steps_scores
      - choose task index by argmax over task_scores (0..l-1); must be < |queue|
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, config: Optional[EnvConfig] = None, seed: Optional[int] = None,
                 enable_eval: bool = False, enable_render: bool = False, eval_csv_path: Optional[str] = None,
                 eval_tasks_csv_path: Optional[str] = None) -> None:
        self.config: EnvConfig = config or get_config()
        self.rng = np.random.default_rng(seed)

        self.max_visible_tasks: int = self.config.max_visible_tasks
        self.dt: float = float(self.config.dt_seconds)
        self.max_sim_seconds: float = float(self.config.max_sim_seconds)

        # Core components
        self.scheduler = NodeScheduler()
        self.generator = TaskGenerator(self.config.prompt_pool, self.config.parallel_choices, rng = self.rng)
        self.task_manager = TaskManager(self.generator)
        
        # 节点管理器（支持仿真/真实模式切换）
        runtime_config = get_runtime_config()
        self.node_manager = NodeManager(
            config=self.config,
            runtime_config=runtime_config,
            mode=runtime_config.mode
        )
        # 保存运行模式用于判断是否需要记录真实时间
        self._is_real_backend = (runtime_config.mode == "real")

        # Dynamic state holders
        self.now_seconds: float = 0.0
        self.queue: Queue = Queue()
        self._running_tasks: List[SDTask] = []

        # Eval / Render controls (not in EnvConfig)
        self._eval_enabled: bool = bool(enable_eval)
        self._render_enabled: bool = bool(enable_render)
        self._eval_csv_path: Optional[str] = str(eval_csv_path) if eval_csv_path else None
        self._eval_csv_header_written: bool = False
        self._eval_tasks_csv_path: Optional[str] = str(eval_tasks_csv_path) if eval_tasks_csv_path else None
        self._eval_tasks_header_written: bool = False
        self._last_action_raw: Optional[np.ndarray] = None
        self._last_action_parsed: Dict[str, Any] = {}
        self._last_completed_latencies: List[float] = []
        self._completed_latency_sum: float = 0.0
        self._last_quality: float = 0.0

        # Gymnasium spaces (optional)
        # Continuous 3D action space: [exec_flag, step_continuous, task_continuous]
        if spaces is not None:
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                shape=(3,),
                dtype=np.float32,
            )

            # Nodes obs space bounds with correct shapes
            node_low_vec = np.array([0.0, 0.0], dtype=np.float32)
            node_high_vec = np.array([np.inf, np.inf], dtype=np.float32)
            node_low = np.tile(node_low_vec, (self.config.num_nodes, 1))
            node_high = np.tile(node_high_vec, (self.config.num_nodes, 1))

            # Queue obs space bounds with correct shapes (2 dims per task)
            queue_low_vec = np.array([0.0, 0.0], dtype=np.float32)
            queue_high_vec = np.array([np.inf, np.inf], dtype=np.float32)
            queue_low = np.tile(queue_low_vec, (self.max_visible_tasks, 1))
            queue_high = np.tile(queue_high_vec, (self.max_visible_tasks, 1))

            if self.config.concat_state:
                # Concatenated observation: shape ((num_nodes*4 + l*3),)
                obs_low = np.concatenate([node_low.flatten(), queue_low.flatten()]).astype(np.float32)
                obs_high = np.concatenate([node_high.flatten(), queue_high.flatten()]).astype(np.float32)
                self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
            else:
                self.observation_space = spaces.Tuple(
                    (
                        spaces.Box(low=node_low, high=node_high, shape=(self.config.num_nodes, 4), dtype=np.float32),
                        spaces.Box(low=queue_low, high=queue_high, shape=(self.max_visible_tasks, 2), dtype=np.float32),
                    )
                )
        self.reset_count = 0

    # Gymnasium API
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
        # print(seed)
        self.reset_count+=1
        # print("env reset")
        
        if seed is not None:
            super().reset(seed=seed) # 调用 gym.Env 的 reset 来处理种子
            self.rng = np.random.default_rng(seed)

        # Init time
        self.now_seconds = 0.0

        self.generator.reset(rng = self.rng)
        
        # 重置节点管理器
        self.node_manager.reset()

        # Init queue empty; arrivals will be generated over time
        self.queue = Queue()
        self._num_generated_tasks: int = 0
        self._num_completed_tasks: int = 0
        self._stats_enabled: bool = getattr(self.config, "enable_statistics", False)
        self._stats_rows: List[Dict[str, Any]] = []
        self._eval_csv_header_written = False
        self._eval_tasks_header_written = False
        self._last_action_raw = None
        self._last_action_parsed = {}
        self._last_completed_latencies = []
        self._running_tasks:List[SDTask] = []
        self._completed_latency_sum = 0.0

        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, bool, Dict[str, Any]]:  # type: ignore[override]
        # Parse continuous 3D action: [exec_flag, step_continuous, task_continuous]
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}
        self._last_action_raw = action.copy()
        self._last_action_parsed = {}

        # action: [exec_flag_cont, step_cont, task_cont], range [-1, 1]
        a_exec = float(action[0])
        a_step = float(action[1])
        a_task = float(action[2])

        # Determine execute/skip
        do_execute = a_exec > 0.0

        if not do_execute:
            self._advance_time_and_arrivals(self.dt)
        else:
            # Map continuous values to valid integers
            # step in [1, max_step]
            max_step = self.config.max_step
            step_scaled = (np.clip(a_step, -1.0, 1.0) + 1.0) * 0.5  # [0,1]
            step_val = int(np.round(1 + step_scaled * (max_step - 1)))

            # task index 1..visible_len
            visible_len = self._visible_queue_len()
            
            task_scaled = (np.clip(a_task, -1.0, 1.0) + 1.0) * 0.5  # [0,1]
            task_1based = int(np.round(1 + task_scaled * (visible_len - 1)))
            task_1based = int(np.clip(task_1based, 1, visible_len))
            task_sel = task_1based - 1  # Convert to 0-based

            if not self._can_execute(task_sel):
                self._advance_time_and_arrivals(self.dt)
            else:
                sd_task = self._build_sd_task_with_steps(task_sel=task_sel, steps=step_val)
                self._apply_schedule(sd_task)
                reward = self._compute_reward(sd_task)

        # Record parsed action info
        self._last_action_parsed = {
            "a_exec": a_exec,
            "a_step": a_step,
            "a_task": a_task,
            "do_execute": bool(a_exec > 0.0),
            "step_val": int(locals().get("step_val", 0)),
            "task_sel": int(locals().get("task_sel", -1)),
        }

        # Check termination
        if self._all_tasks_completed():
            terminated = True
        if self.now_seconds >= self.max_sim_seconds:
            truncated = True

        # 完成检测已在 _advance_time / _check_task_completion 中进行
        # 将最近一次完成的任务统计放入 info（供收集器或外部评估使用）
        # 为了与 Tianshou 的 VectorReplayBuffer 兼容，避免 variable-length 列表导致形状推断为 0
        # 仅返回聚合指标：完成数量与总响应时间和
        cnt = int(len(self._last_completed_latencies))
        ssum = float(sum(self._last_completed_latencies)) if cnt > 0 else 0.0
        info["completed_count"] = cnt
        info["completed_latency_sum"] = ssum
        info["quality"] = self._last_quality

        if self.reset_count == 8:
            assert "check cnt"

        # 评估模式：每步写入 CSV，且可渲染
        if self._eval_enabled:
            try:
                self._log_step_csv()
            except Exception:
                pass
            if self._render_enabled:
                self._do_render_and_wait()

        obs = self._get_obs()
        return obs, float(reward), bool(terminated), bool(truncated), info

    # Helpers
    def _get_obs(self):
        nodes_state = self.node_manager.get_states()
        queue_state = self.queue.as_state(self.max_visible_tasks).astype(np.float32)
        if self.config.concat_state:
            return np.concatenate([nodes_state.flatten(), queue_state.flatten()]).astype(np.float32)
        return nodes_state, queue_state


    def _visible_queue_len(self) -> int:
        return self.max_visible_tasks

    def _idle_node_indices(self) -> List[int]:
        return self.node_manager.get_idle_node_indices()

    def _can_execute(self, task_sel: int) -> bool:
        # Queue must have items and selection must be in range of visible tasks
        if len(self.queue) == 0:
            return False
        if task_sel < 0 or task_sel >= len(self.queue):
            return False
        user_task = self.queue.peek(task_sel)
        if user_task is None:
            return False
        # Legality: available nodes >= required parallelism
        available = len(self._idle_node_indices())
        return available >= user_task.parallelism

    def _estimate_exec_seconds(self, steps: int, parallelism: int) -> float:
        base = self.config.time_pred_a * float(steps) + self.config.time_pred_b
        if self.config.linear_parallel_speedup and parallelism > 1:
            base = base / float(parallelism)
        base = max(0.0, base)
        return base


    def _build_sd_task_with_steps(self, task_sel: int, steps: int) -> SDTask:
        # Build SDTask with given steps value
        user_task = self.queue.pop(task_sel)
        assert user_task is not None

        # Decide nodes via scheduler
        required = user_task.parallelism
        # 创建临时节点列表用于调度
        temp_nodes = []
        for i in range(self.node_manager.num_nodes):
            temp_node = Node(
                index=i,
                model_id=self.node_manager.get_node_model_id(i),
                remaining_seconds=self.node_manager.get_node_remaining_time(i),
                compute_power=self.config.compute_power[i]
            )
            temp_nodes.append(temp_node)
        
        chosen = self.scheduler.select_nodes(temp_nodes, required)
        if not chosen or len(chosen) != required:
            self._reinsert_task_at(user_task, task_sel)
            raise RuntimeError("Node scheduling failed despite legality check")

        reload_required = not self.node_manager.is_exact_group(chosen)
        
        # Calculate minimum compute power among assigned nodes (heterogeneous)
        min_compute_power = min(self.config.compute_power[i] for i in chosen)

        sd_task = SDTask(
            assigned_node_indices=chosen,
            steps=int(steps),
            reload_required=reload_required,
            prompt=user_task.prompt,
            arrival_time=user_task.arrival_time,
            min_compute_power=min_compute_power,
            real_arrival_time=user_task.real_arrival_time,
        )
        return sd_task

    def _reinsert_task_at(self, task: UserTask, index: int) -> None:
        # Reinsert maintaining order; used only on failure
        # Simple: push to end if index out of range
        if index < 0 or index > len(self.queue):
            self.queue.push(task)
            return
        # rebuild items
        from .task import Queue as _Q
        items: List[UserTask] = []
        # Extract existing
        for i in range(len(self.queue)):
            t = self.queue.pop(0)
            if t is not None:
                items.append(t)
        # Insert back
        items.insert(min(index, len(items)), task)
        self.queue = _Q()
        for t in items:
            self.queue.push(t)

    def _is_exact_group(self, chosen: List[int]) -> bool:
        return self.node_manager.is_exact_group(chosen)

    def _apply_schedule(self, sd_task: SDTask) -> None:
        chosen = sd_task.assigned_node_indices
        
        # 先分配任务（真实模式会在这里做destroy/init）
        # 必须在修改model_id之前调用，这样backend才能正确计算affected_groups
        self.node_manager.assign_task(sd_task)
        
        # Model group update logic（仅用于仿真模式和本地状态跟踪）
        if sd_task.reload_required:
            # Unload all involved model groups to 0, then load new model group at chosen nodes
            self.node_manager.unload_model_groups(chosen)
            # Assign a new temporary group id as (max_existing + 1)
            new_group_id = self.node_manager.get_max_group_id() + 1
            self.node_manager.load_model_group(chosen, new_group_id)
        else:
            # Reuse exact group: nothing to change in model ids
            pass

        # Renumber model groups each frame to keep ids compact starting from 1
        self.node_manager.renumber_groups()

        # Track running task for completion monitoring
        self._running_tasks.append(sd_task)

    def _renumber_groups(self) -> None:
        self.node_manager.renumber_groups()

    def _advance_time(self, dt: float) -> None:
        # Advance simulation clock and update nodes, also renumber groups
        self.now_seconds = float(self.now_seconds + dt)
        self.node_manager.update(dt=dt, now=self.now_seconds)
        self._check_task_completion()
        self.node_manager.renumber_groups()

    def _advance_time_and_arrivals(self, dt: float) -> None:
        # Generate arrivals during (now, now+dt], then advance time
        lam = self.config.poisson_lambda_per_sec
        arrivals = self.task_manager._generator.poisson_arrivals(lam_per_sec=lam, dt_seconds=dt, now=self.now_seconds)
        for t in arrivals:
            # Stop when reaching configured total tasks
            if self._num_generated_tasks >= self.config.total_tasks:
                break
            # 如果是真实后端，记录任务的真实到达时间
            if self._is_real_backend:
                import time
                t.real_arrival_time = time.time()
            self.queue.push(t)
            self._num_generated_tasks += 1
        self._advance_time(dt)

    def _all_tasks_completed(self) -> bool:
        # Done when generated == total and queue empty and no running tasks remain
        if self._num_generated_tasks < self.config.total_tasks:
            return False
        if len(self.queue) > 0:
            return False
        for i in range(self.node_manager.num_nodes):
            if self.node_manager.get_node_remaining_time(i) > 0.0:
                return False
        return True

    def _check_task_completion(self) -> None:
        if not self._running_tasks:
            return
        still_running: List[SDTask] = []
        # reset last completed list for this frame
        self._last_completed_latencies = []
        for task in self._running_tasks:
            done = True
            for idx in task.assigned_node_indices:
                if self.node_manager.get_node_remaining_time(idx) > 0.5:
                    done = False
                    break
            if done:
                task.completion_time = self.now_seconds
                # 如果是真实后端，记录任务的真实完成时间
                if self._is_real_backend:
                    import time
                    task.real_completion_time = time.time()
                self._num_completed_tasks += 1
                if self._stats_enabled:
                    self._log_task_stat(task)
                # 记录该任务的响应时间
                latency = float((task.completion_time) - float(task.arrival_time))
                if latency<0:
                    assert "time<0 error!"
                self._last_completed_latencies.append(latency)
                self._completed_latency_sum = float(self._completed_latency_sum + latency)
                # eval 模式导出任务级数据
                if self._eval_enabled:
                    try:
                        self._log_completed_task_csv(task=task, latency=latency)
                    except Exception:
                        pass
            else:
                still_running.append(task)
                
        self._running_tasks = still_running

    def _log_task_stat(self, task: SDTask) -> None:
        path = getattr(self.config, "stats_csv_path", "env_stats.csv")
        file_exists = os.path.exists(path)
        row = {
            "arrival_time": float(task.arrival_time),
            "completion_time": float(task.completion_time or self.now_seconds),
            "steps": int(task.steps),
            "parallelism": int(len(task.assigned_node_indices)),
        }
        with open(path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["arrival_time", "completion_time", "steps", "parallelism"])
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    # Reward with image transfer time in t1 term
    def _compute_reward(self, task: SDTask) -> float:
        # t1: scheduled task duration (load+exec approximated via node timers) + image transfer time
        if not task.assigned_node_indices:
            return 0.0

        # Job time as max remaining among assigned nodes (after receive)
        job_seconds = 0.0
        for idx in task.assigned_node_indices:
            job_seconds = max(job_seconds, float(self.node_manager.get_node_remaining_time(idx)))

        # Approximate bandwidth at admission as the mean instantaneous bandwidth of chosen nodes
        # 注意：真实模式下，带宽信息需要从API获取，这里使用配置默认值
        bandwidth_values: List[float] = []
        for idx in task.assigned_node_indices:
            # 仿真模式：使用节点的网络状态；真实模式：使用配置的默认带宽
            # 这里简化处理，使用配置中的默认带宽
            node_type = self.config.node_types[idx]
            # 假设使用good状态的带宽
            bw = self.config.bandwidth_by_state.get("good", 100.0)
            bandwidth_values.append(float(bw))
        avg_bw = float(np.mean(bandwidth_values)) if len(bandwidth_values) > 0 else 0.0
        image_size_mb = float(getattr(self.config, "image_size_mb", 3.0))
        transfer_seconds = image_size_mb / max(1e-9, avg_bw)

        t1 = job_seconds + transfer_seconds

        # t2: average waiting time for tasks currently in queue
        queue_wait_times: List[float] = []
        if len(self.queue) > 0:
            for i in range(len(self.queue)):
                ut = self.queue.peek(i)
                if ut is None:
                    continue
                wait_time = max(0.0, float(self.now_seconds - ut.arrival_time))
                queue_wait_times.append(wait_time)
        t2_avg = float(np.mean(queue_wait_times)) if queue_wait_times else 0.0

        # q_raw: raw quality metric from task steps (benefit)
        q_raw = float(task.steps)

        # 1. Calculate normalized costs for each component using tanh
        # Cost for t1 (single-task latency)
        cost_t1 = np.tanh(t1 / float(self.config.reward_t1_scale))

        # Cost for t2 (queue latency)
        # If queue is empty, t2_avg is 0, so cost is tanh(0) = 0. This is a sensible default.
        cost_t2 = np.tanh(t2_avg / float(self.config.reward_t2_scale))

        # Cost for quality (benefit converted to cost)
        # We use the '1 - tanh(benefit)' form
        quality = np.tanh(q_raw / float(self.config.reward_q_scale))

        # Store quality for metrics tracking
        self._last_quality = float(quality)

        # 2. Calculate total weighted cost
        total_cost = (
            float(self.config.reward_t1_weight) * cost_t1 +
            float(self.config.reward_t2_weight) * cost_t2
        )
        
        reward = -total_cost + float(self.config.reward_q_weight) * quality + self.config.reward_bais

        if reward <= 0:
            print()

        assert reward > 0, "forbiddened negetive or 0 reward"

        return float(reward)

    def _compute_quality(self, steps: int) -> float:
        a = float(getattr(self.config, "reward_quality_a", 1.0))
        b = float(getattr(self.config, "reward_quality_b", 1.0))
        c = float(getattr(self.config, "reward_quality_c", 1.0))
        return np.minimum(float(np.log(steps)/np.log(6)), c)

    def preview_reward(self, action: np.ndarray) -> float:
        """Preview reward for an action without executing it.
        
        Args:
            action: 3D continuous action [exec_flag, step_continuous, task_continuous]
            
        Returns:
            Expected reward for this action (0.0 if illegal)
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        
        # Parse action
        a_exec = float(action[0])
        a_step = float(action[1])
        a_task = float(action[2])
        
        # If skip, reward is 0
        if a_exec <= 0.0:
            return 0.0
        
        # Map continuous values to integers
        max_step = self.config.max_step
        step_scaled = (np.clip(a_step, -1.0, 1.0) + 1.0) * 0.5
        step_val = int(np.round(1 + step_scaled * (max_step - 1)))
        
        visible_len = self._visible_queue_len()
        task_scaled = (np.clip(a_task, -1.0, 1.0) + 1.0) * 0.5
        task_1based = int(np.round(1 + task_scaled * (visible_len - 1)))
        task_1based = int(np.clip(task_1based, 1, visible_len))
        task_sel = task_1based - 1
        
        # Check legality
        if not self._can_execute(task_sel):
            return 0.0
        
        # Peek task (don't pop)
        user_task = self.queue.peek(task_sel)
        if user_task is None:
            return 0.0
        
        # Simulate node selection
        required = user_task.parallelism
        # 创建临时节点列表用于调度（仅用于预览）
        temp_nodes = []
        for i in range(self.node_manager.num_nodes):
            temp_node = Node(
                index=i,
                model_id=self.node_manager.get_node_model_id(i),
                remaining_seconds=self.node_manager.get_node_remaining_time(i),
                compute_power=self.config.compute_power[i]
            )
            temp_nodes.append(temp_node)
        
        chosen = self.scheduler.select_nodes(temp_nodes, required)
        if not chosen or len(chosen) != required:
            return 0.0
        
        # Calculate minimum compute power
        min_compute_power = min(self.config.compute_power[i] for i in chosen)
        
        # Determine reload requirement
        reload_required = not self.node_manager.is_exact_group(chosen)
        
        # Create temporary SDTask for reward calculation
        temp_task = SDTask(
            assigned_node_indices=chosen,
            steps=int(step_val),
            reload_required=reload_required,
            prompt=user_task.prompt,
            arrival_time=user_task.arrival_time,
            min_compute_power=min_compute_power,
        )
        
        # Simulate the task being received by nodes (for reward calculation)
        # We need to calculate what the remaining_seconds would be
        parallelism = len(chosen)
        base_exec = self.config.time_pred_a * float(step_val) + self.config.time_pred_b
        weight = self.config.parallel_speedup_weights.get(parallelism)
        if weight is None:
            weight = float(parallelism) if self.config.linear_parallel_speedup else 1.0
        base_exec = base_exec / max(1e-9, weight)
        
        load_seconds = 0.0
        if reload_required:
            load_seconds = self.config.model_load_seconds_by_parallelism.get(
                parallelism, self.config.model_load_seconds_unified
            )
        
        simulated_job_seconds = load_seconds + (base_exec / max(1e-9, min_compute_power))
        
        # Calculate reward components (similar to _compute_reward but without modifying state)
        bandwidth_values: List[float] = []
        for idx in chosen:
            # 使用配置的默认带宽
            bw = self.config.bandwidth_by_state.get("good", 100.0)
            bandwidth_values.append(float(bw))
        avg_bw = float(np.mean(bandwidth_values)) if bandwidth_values else 0.0
        image_size_mb = float(self.config.image_size_mb)
        transfer_seconds = image_size_mb / max(1e-9, avg_bw)
        
        t1 = simulated_job_seconds + transfer_seconds
        
        # t2: average waiting time for tasks in queue (excluding the selected task)
        queue_wait_times: List[float] = []
        for i in range(len(self.queue)):
            if i == task_sel:  # Skip the task we're selecting
                continue
            ut = self.queue.peek(i)
            if ut is None:
                continue
            wait_time = max(0.0, float(self.now_seconds - ut.arrival_time))
            queue_wait_times.append(wait_time)
        t2_avg = float(np.mean(queue_wait_times)) if queue_wait_times else 0.0
        
        # Quality
        q_raw = float(step_val)
        
        # Calculate reward
        cost_t1 = np.tanh(t1 / float(self.config.reward_t1_scale))
        cost_t2 = np.tanh(t2_avg / float(self.config.reward_t2_scale))
        quality = np.tanh(q_raw / float(self.config.reward_q_scale))
        
        total_cost = (
            float(self.config.reward_t1_weight) * cost_t1 +
            float(self.config.reward_t2_weight) * cost_t2
        )
        
        reward = -total_cost + float(self.config.reward_q_weight) * quality + self.config.reward_bais
        
        return float(max(0.0, reward))

    # ===== Eval / Render helpers =====
    def _collect_step_metrics(self) -> Dict[str, Any]:
        # per-node remaining times
        node_remaining: List[float] = [
            float(self.node_manager.get_node_remaining_time(i))
            for i in range(self.node_manager.num_nodes)
        ]
        # queue metrics
        q_len = int(len(self.queue))
        waits: List[float] = []
        for i in range(q_len):
            ut = self.queue.peek(i)
            if ut is not None:
                waits.append(max(0.0, float(self.now_seconds - float(ut.arrival_time))))
        avg_wait = float(np.mean(waits)) if len(waits) > 0 else 0.0
        # first l tasks (parallelism and arrival time)
        l = int(self.max_visible_tasks)
        tasks_info: List[Tuple[int, float]] = []
        upto = min(l, q_len)
        for i in range(upto):
            ut = self.queue.peek(i)
            if ut is not None:
                tasks_info.append((int(ut.parallelism), float(ut.arrival_time)))
        # pad to l
        while len(tasks_info) < l:
            tasks_info.append((0, 0.0))
        return {
            "node_remaining": node_remaining,
            "queue_len": q_len,
            "queue_avg_wait": avg_wait,
            "tasks_info": tasks_info,
        }

    def _ensure_eval_csv_header(self) -> None:
        if not self._eval_csv_path or self._eval_csv_header_written:
            return
        # build header
        headers: List[str] = []
        for i in range(int(self.config.num_nodes)):
            headers.append(f"node_{i}_remaining_seconds")
        headers += ["queue_len", "queue_avg_wait_seconds"]
        for i in range(int(self.max_visible_tasks)):
            headers.append(f"task{i+1}_parallelism")
            headers.append(f"task{i+1}_arrival_time")
        # write header
        os.makedirs(os.path.dirname(self._eval_csv_path) or ".", exist_ok=True)
        file_exists = os.path.exists(self._eval_csv_path)
        with open(self._eval_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
        self._eval_csv_header_written = True

    def _log_step_csv(self) -> None:
        if not self._eval_csv_path:
            return
        self._ensure_eval_csv_header()
        metrics = self._collect_step_metrics()
        row: List[Any] = []
        row.extend(metrics["node_remaining"])  # per-node remaining
        row.append(metrics["queue_len"])  # total tasks
        row.append(metrics["queue_avg_wait"])  # avg wait
        for (co, at) in metrics["tasks_info"]:
            row.append(co)
            row.append(at)
        with open(self._eval_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _do_render_and_wait(self) -> None:
        # print node states
        print("==== EdgeEnv Render ====")
        print(f"time: {self.now_seconds:.3f}s, queue_len: {len(self.queue)}")
        # Nodes summary
        nodes_summary = ", ".join([
            f"n{i}: remain={self.node_manager.get_node_remaining_time(i):.2f}, gid={self.node_manager.get_node_model_id(i)}"
            for i in range(self.node_manager.num_nodes)
        ])
        print(f"nodes: {nodes_summary}")
        # Queue summary (first l)
        upto = min(int(self.max_visible_tasks), len(self.queue))
        q_items: List[str] = []
        for i in range(upto):
            ut = self.queue.peek(i)
            if ut:
                q_items.append(f"[{i}] p={ut.parallelism}, at={ut.arrival_time:.2f}")
        print("queue(head): " + ("; ".join(q_items) if q_items else "<empty>"))
        # Action summary
        if self._last_action_raw is not None:
            print(f"last_action_raw: {np.array2string(self._last_action_raw, precision=3)}")
        if self._last_action_parsed:
            print(f"parsed: {self._last_action_parsed}")
        try:
            input("[render] Press Enter to continue...")
        except EOFError:
            pass

    # ======= Eval tasks CSV helpers & public stats =======
    def _ensure_eval_tasks_header(self) -> None:
        if self._eval_tasks_header_written:
            return
        # derive default path from eval_csv_path if not provided
        if not self._eval_tasks_csv_path:
            base_dir = os.path.dirname(self._eval_csv_path) if self._eval_csv_path else "."
            self._eval_tasks_csv_path = os.path.join(base_dir, "eval_tasks.csv")
        os.makedirs(os.path.dirname(self._eval_tasks_csv_path) or ".", exist_ok=True)
        file_exists = os.path.exists(self._eval_tasks_csv_path)
        with open(self._eval_tasks_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                # 基础列
                header = ["arrival_time", "parallelism", "steps", "spent_time", "completion_time", "node_indices"]
                # 如果是真实后端，添加真实时间列
                if self._is_real_backend:
                    header.extend(["real_arrival_time", "real_completion_time", "real_spent_time"])
                writer.writerow(header)
        self._eval_tasks_header_written = True

    def _log_completed_task_csv(self, task: SDTask, latency: float) -> None:
        self._ensure_eval_tasks_header()
        if not self._eval_tasks_csv_path:
            return
        
        # 基础数据
        row = [
            float(task.arrival_time),
            int(len(task.assigned_node_indices)),
            int(task.steps),
            float(latency),
            float(task.completion_time),
            ",".join(map(str, task.assigned_node_indices)),  # 节点编号，用逗号分隔
        ]
        
        # 如果是真实后端，添加真实时间数据
        if self._is_real_backend:
            real_arrival = float(task.real_arrival_time) if task.real_arrival_time is not None else 0.0
            real_completion = float(task.real_completion_time) if task.real_completion_time is not None else 0.0
            real_latency = real_completion - real_arrival if (task.real_arrival_time is not None and task.real_completion_time is not None) else 0.0
            row.extend([real_arrival, real_completion, real_latency])
        
        # check latency = completion_time - arrival_time
        assert abs(latency - float(task.completion_time - task.arrival_time)) <= 1, \
        f"Latency error! Passed: {latency}, Calculated: {float(task.completion_time - task.arrival_time)}, Error: {latency - float(task.completion_time - task.arrival_time)}"
        with open(self._eval_tasks_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def get_completed_stats(self) -> Tuple[int, float]:
        """Return (completed_count, average_response_time_seconds)."""
        count = int(self._num_completed_tasks)
        avg = float(self._completed_latency_sum / max(1, count)) if count > 0 else 0.0
        return count, avg

def make_env(config: Optional[EnvConfig] = None, seed: Optional[int] = None):
    def _factory():
        return EdgeEnv(config=config, seed=seed)
    # Try to return tianshou DummyVectorEnv with a single env
    try:  # pragma: no cover - optional dependency
        from tianshou.env import DummyVectorEnv

        return DummyVectorEnv([_factory])
    except Exception:
        # Fallback: return raw env instance for interactive usage
        return _factory()


