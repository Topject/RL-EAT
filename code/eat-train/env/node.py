from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import numpy as np
import random
from .task import SDTask


@dataclass
class Node:
    index: int
    model_id: int = 0  # 0 means no model loaded
    compute_power: float = 1.0
    model_load_seconds: float = 8.0
    # Optional: load time per task parallelism
    model_load_seconds_by_parallelism: Dict[int, float] = field(default_factory=dict)
    # time prediction parameters a*steps + b
    time_pred_a: float = 0.10
    time_pred_b: float = 1.0
    linear_parallel_speedup: bool = True
    node_type: str = "edge"  # "edge" or "cluster"
    network_states: Tuple[str, str] = ("good", "normal")
    network_transition: Dict[str, List[List[float]]] = field(default_factory=dict)
    bandwidth_by_state: Dict[str, float] = field(default_factory=dict)
    bandwidth_window_seconds: float = 10.0
    # Parallel speedup weights mapping: exec_time /= weight[parallelism]
    parallel_speedup_weights: Dict[int, float] = field(default_factory=dict)

    # dynamic
    remaining_seconds: float = 0.0
    current_network_state: str = "good"
    _bandwidth_samples: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, bandwidth)

    def is_idle(self) -> bool:
        return self.remaining_seconds <= 0.0

    def update(self, dt: float, now: float) -> None:
        # Update task remaining time
        if self.remaining_seconds > 0.0:
            self.remaining_seconds = max(0.0, self.remaining_seconds - dt)

        # Update network state via Markov transition
        transitions = self.network_transition.get(self.node_type)
        if transitions is not None:
            states = list(self.network_states)
            idx = states.index(self.current_network_state)
            probs = transitions[idx]
            next_idx = 0 if random.random() < probs[0] else 1
            self.current_network_state = states[next_idx]

        # Record bandwidth sample
        bw = self.bandwidth_by_state.get(self.current_network_state, 0.0)
        self._bandwidth_samples.append((now, bw))
        # Drop old samples
        cutoff = now - self.bandwidth_window_seconds
        self._bandwidth_samples = [(t, b) for (t, b) in self._bandwidth_samples if t >= cutoff]

    def state(self) -> np.ndarray:
        # [remaining_time, model_id, avg_bandwidth, std_bandwidth]
        bw_values = [b for (_, b) in self._bandwidth_samples]
        if not bw_values:
            avg = 0.0
            std = 0.0
        else:
            avg = float(np.mean(bw_values))
            std = float(np.std(bw_values))

        return np.array([self.remaining_seconds/30, float(self.model_id)/30], dtype=np.float32)
        # return np.array([self.remaining_seconds/30, float(self.model_id)/30, avg/100, std], dtype=np.float32)

    def receive(self, sd_task: SDTask) -> Optional[SDTask]:
        # Accept task if idle; compute duration = load + exec
        if not self.is_idle():
            return None
        base_exec = self.time_pred_a * float(sd_task.steps) + self.time_pred_b
        parallelism = max(1, len(sd_task.assigned_node_indices))
        # Prefer explicit weights; otherwise fall back to linear if enabled
        weight = self.parallel_speedup_weights.get(parallelism)
        if weight is None:
            weight = float(parallelism) if self.linear_parallel_speedup else 1.0
        base_exec = base_exec / max(1e-9, weight)
        # Choose load time: per-parallelism override if provided
        load_seconds = 0.0
        if sd_task.reload_required:
            load_seconds = self.model_load_seconds_by_parallelism.get(parallelism, self.model_load_seconds)
        # Use minimum compute power among assigned nodes (heterogeneous computing)
        effective_compute_power = sd_task.min_compute_power
        total_seconds = load_seconds + (
            base_exec / max(1e-9, effective_compute_power)
        )
        self.remaining_seconds = max(0.0, total_seconds)
        return sd_task


