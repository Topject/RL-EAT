from __future__ import annotations

import os
import time
import uuid
from typing import List, Dict, Set, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import threading
import numpy as np
import requests
from .base import Backend
from ..task import SDTask


class TaskManager:
    """任务管理器 - 负责真实环境的HTTP请求管理"""
    
    def __init__(
        self,
        node_ips: List[str],
        runtime_config,
        timeout: float = 30.0
    ):
        """初始化任务管理器
        
        Args:
            node_ips: 节点IP地址列表
            runtime_config: 运行时配置对象
            timeout: API请求超时时间
        """
        self.node_ips = node_ips
        self.runtime_config = runtime_config
        self.timeout = timeout
        self.current_job_id: Optional[str] = None
        
        # 后台draw任务管理
        self.draw_executor = ThreadPoolExecutor(max_workers=10)
        self.active_draw_futures: Dict[str, List[Future]] = {}  # job_id -> futures
        self.draw_results: Dict[str, Optional[bytes]] = {}  # job_id -> image_data
        
        # 端口管理：维护可用端口池
        self.port_pool: Set[int] = set(range(29500, 29600))  # 100个端口
        self.port_lock = threading.Lock()
        self.job_ports: Dict[str, int] = {}  # job_id -> allocated_port
        
        # 确保输出目录存在
        os.makedirs(runtime_config.output_dir, exist_ok=True)
    
    def _allocate_port(self, job_id: str) -> int:
        """为任务分配一个可用端口
        
        Args:
            job_id: 任务ID
            
        Returns:
            分配的端口号
            
        Raises:
            RuntimeError: 无可用端口
        """
        with self.port_lock:
            if not self.port_pool:
                raise RuntimeError("端口池已耗尽，无法分配新端口")
            
            port = self.port_pool.pop()
            self.job_ports[job_id] = port
            self._debug_log(f"为任务 {job_id} 分配端口 {port}，剩余可用端口: {len(self.port_pool)}")
            return port
    
    def _release_port(self, job_id: str) -> None:
        """释放任务占用的端口
        
        Args:
            job_id: 任务ID
        """
        with self.port_lock:
            if job_id in self.job_ports:
                port = self.job_ports.pop(job_id)
                self.port_pool.add(port)
                self._debug_log(f"释放任务 {job_id} 的端口 {port}，剩余可用端口: {len(self.port_pool)}")
    
    def _debug_log(self, message: str) -> None:
        """输出调试信息"""
        if self.runtime_config.debug:
            print(f"[DEBUG] {message}")
    
    def _post_request(self, node_ip: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """发送POST请求
        
        Args:
            node_ip: 节点IP地址
            endpoint: API端点
            data: 请求数据
            
        Returns:
            响应JSON数据
        """
        url = f"http://{node_ip}{endpoint}"
        self._debug_log(f"POST {url} with data={data}")
        try:
            response = requests.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            self._debug_log(f"Response from {url}: {result}")
            return result
        except Exception as e:
            error_msg = f"请求失败 {url}: {e}"
            print(error_msg)
            return {"error": str(e)}
    
    def _get_request(self, node_ip: str, endpoint: str) -> Dict:
        """发送GET请求
        
        Args:
            node_ip: 节点IP地址
            endpoint: API端点
            
        Returns:
            响应JSON数据
        """
        url = f"http://{node_ip}{endpoint}"
        # self._debug_log(f"GET {url}")
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            # self._debug_log(f"Response from {url}: {result}")
            return result
        except Exception as e:
            print(f"请求失败 {url}: {e}")
            return {"error": str(e)}
    
    def destroy_node(self, node_index: int) -> bool:
        """销毁节点上的模型
        
        Args:
            node_index: 节点索引
            
        Returns:
            是否成功
        """
        if node_index >= len(self.node_ips):
            return False
        
        node_ip = self.node_ips[node_index]
        result = self._post_request(node_ip, "/destroy", None)
        return "error" not in result
    
    def init_node(
        self,
        node_index: int,
        job_id: str,
        rank: int,
        world_size: int,
        local_rank: int,
        master_addr: str,
        master_port: int
    ) -> bool:
        """初始化节点
        
        Args:
            node_index: 节点索引
            job_id: 任务ID
            rank: 全局rank
            world_size: 总进程数
            local_rank: 本地rank
            master_port: 分布式训练主端口
            
        Returns:
            是否成功
        """
        if node_index >= len(self.node_ips):
            return False
        
        node_ip = self.node_ips[node_index]
        cfg = self.runtime_config
        
        data = {
            "job_id": job_id,
            "master_addr": master_addr,
            "master_port": master_port,
            "world_size": world_size,
            "rank": rank,
            "local_rank": local_rank,
            "height": cfg.height,
            "width": cfg.width,
            "warmup_steps": cfg.warmup_steps,
            "mode": cfg.backend_mode,
            "parallelism": cfg.parallelism,
            "split_batch": cfg.split_batch,
            "use_cuda_graph": cfg.use_cuda_graph,
            "guidance_scale": cfg.guidance_scale,
        }
        
        result = self._post_request(node_ip, "/init", data)
        return "error" not in result
    
    def draw(
        self,
        node_index: int,
        job_id: str,
        prompt: str,
        steps: int,
        seed: Optional[int] = None,
        rank: int = 0
    ) -> Tuple[bool, Optional[bytes]]:
        """在节点上执行绘图任务（阻塞直到完成）
        
        Args:
            node_index: 节点索引
            job_id: 任务ID
            prompt: 提示词
            steps: 推理步数
            seed: 随机种子
            rank: 节点的rank（0表示主节点，会返回图像）
            
        Returns:
            (是否成功, 图像数据) - 只有rank 0返回图像数据
        """
        if node_index >= len(self.node_ips):
            return False, None
        
        node_ip = self.node_ips[node_index]
        cfg = self.runtime_config
        
        data = {
            "job_id": job_id,
            "prompt": prompt,
            "num_inference_steps": steps,
            "guidance_scale": cfg.guidance_scale,
            "seed": seed or cfg.seed,
        }
        
        url = f"http://{node_ip}/draw"
        self._debug_log(f"POST node{node_index}(rank={rank}) {url} with data={data}")
        try:
            # draw请求会阻塞直到完成
            response = requests.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            
            # rank 0应该返回图像
            if rank == 0:
                if response.headers.get('Content-Type', '').startswith('image/'):
                    self._debug_log(f"node{node_index}(rank=0) draw completed, received image ({len(response.content)} bytes)")
                    return True, response.content
                else:
                    print(f"警告: rank 0节点 {node_index} 没有返回图像")
                    return False, None
            else:
                # 非rank 0节点只需要检查是否成功（可能返回JSON或空）
                self._debug_log(f"node{node_index}(rank={rank}) draw completed successfully")
                return True, None
                
        except requests.Timeout as e:
            print(f"节点 {node_index} 绘图超时 ({self.timeout}s): {e}")
            return False, None
        except Exception as e:
            print(f"节点 {node_index} 绘图请求失败 {url}: {e}")
            return False, None
    
    def get_remaining_time(self, node_index: int) -> float:
        """获取节点剩余执行时间
        
        Args:
            node_index: 节点索引
            
        Returns:
            剩余时间（秒）
        """
        if node_index >= len(self.node_ips):
            return 0.0
        
        node_ip = self.node_ips[node_index]
        try:
            result = self._get_request(node_ip, "/remain_time")
            return float(result.get("remain_time", 0.0))
        except Exception as e:
            # remain_time查询失败不应该是致命错误
            # print(f"查询节点 {node_index} 剩余时间失败: {e}")
            return 0.0
    
    def execute_task(
        self,
        task: SDTask,
        affected_groups: List[List[int]]
    ) -> bool:
        """执行完整的任务流程：destroy → init → draw
        
        Args:
            task: SD任务对象
            affected_groups: 受影响的进程组列表（需要销毁的节点组）
            
        Returns:
            是否执行成功
        """
        # Step a: 销毁受影响的进程组（真正并发，所有节点同时接收）
        # 按进程组分组处理，确保同组节点同时发送destroy
        for group_nodes in affected_groups:
            if not group_nodes:
                continue
            
            self._debug_log(f"销毁进程组: 节点 {group_nodes}")
            with ThreadPoolExecutor(max_workers=len(group_nodes)) as executor:
                # 同时提交所有destroy请求
                futures = [executor.submit(self.destroy_node, idx) for idx in group_nodes]
                # 等待所有请求完成（barrier同步）
                for idx, future in zip(group_nodes, futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"销毁节点 {idx} 时出错: {e}")
        
        # Step b: 初始化新进程组（并发执行以提高速度）
        job_id = str(uuid.uuid4())
        self.current_job_id = job_id
        
        assigned_nodes = task.assigned_node_indices
        world_size = len(assigned_nodes)
        
        # 为这个任务分配一个唯一的master_port
        # node_ips的格式为 ip:port
        master_addr = self.node_ips[assigned_nodes[0]][:-5]
        try:
            master_port = self._allocate_port(job_id)
        except RuntimeError as e:
            print(f"端口分配失败: {e}")
            return False
        
        # 同时向所有节点发送init（barrier同步）
        self._debug_log(f"初始化进程组: 节点 {assigned_nodes}, master_addr={master_addr}, master_port={master_port}")
        init_success = True
        with ThreadPoolExecutor(max_workers=world_size) as executor:
            # 同时提交所有init请求
            futures = [
                executor.submit(
                    self.init_node,
                    node_idx, job_id, rank, world_size, rank, master_addr, master_port
                )
                for rank, node_idx in enumerate(assigned_nodes)
            ]
            # 等待所有请求完成（barrier同步）
            for node_idx, future in zip(assigned_nodes, futures):
                try:
                    if not future.result():
                        print(f"初始化节点 {node_idx} 失败")
                        init_success = False
                except Exception as e:
                    print(f"初始化节点 {node_idx} 时出错: {e}")
                    init_success = False
        
        if not init_success:
            # 初始化失败，释放端口
            self._release_port(job_id)
            return False
        
        # Step c: 异步执行绘图任务（所有节点同时接收，后台执行）
        # 注意：所有节点都要同时发送draw请求（barrier同步）
        # draw会在后台线程执行，不阻塞返回
        self._debug_log(f"开始异步draw任务 job_id={job_id}, nodes={assigned_nodes}, 使用同步发送")
        
        # 在后台线程中同时发送所有draw请求
        def _send_and_save_results():
            self._debug_log(f"后台线程开始，同时发送所有draw请求...")
            
            # 使用线程池同时发送所有draw请求（barrier同步）
            draw_futures = []
            with ThreadPoolExecutor(max_workers=world_size) as sync_executor:
                # 同时提交所有draw请求
                for rank, node_idx in enumerate(assigned_nodes):
                    future = sync_executor.submit(
                        self.draw,
                        node_idx, job_id, task.prompt, task.steps, self.runtime_config.seed, rank
                    )
                    draw_futures.append((rank, node_idx, future))
                
                # 保存futures
                self.active_draw_futures[job_id] = [f for _, _, f in draw_futures]
            
            # 等待所有draw完成
            self._debug_log(f"所有draw请求已发送，等待完成...")
            image_data = None
            draw_success = True
            completed_nodes = 0
            
            for rank, node_idx, future in draw_futures:
                try:
                    self._debug_log(f"等待节点 {node_idx}(rank={rank}) 完成...")
                    success, img_data = future.result(timeout=None)  # 等待完成，无额外超时
                    completed_nodes += 1
                    
                    if not success:
                        draw_success = False
                        print(f"节点 {node_idx}(rank={rank}) 绘图失败")
                    elif rank == 0:
                        # 只保存rank 0的图像数据
                        image_data = img_data
                        if img_data is None:
                            draw_success = False
                            print(f"节点 {node_idx}(rank=0) 没有返回图像数据")
                    else:
                        self._debug_log(f"节点 {node_idx}(rank={rank}) 完成")
                    
                except Exception as e:
                    draw_success = False
                    print(f"节点 {node_idx}(rank={rank}) 绘图时出错: {e}")
                    import traceback
                    traceback.print_exc()
            
            self._debug_log(f"任务 {job_id} 所有节点完成 ({completed_nodes}/{len(draw_futures)})")
            
            # 保存结果
            if draw_success and image_data:
                timestamp = int(time.time() * 1000)
                filename = f"result_{job_id[:8]}_{timestamp}.png"
                filepath = os.path.join(self.runtime_config.output_dir, filename)
                
                try:
                    with open(filepath, 'wb') as f:
                        f.write(image_data)
                    self._debug_log(f"任务完成，结果保存至: {filepath}")
                    self.draw_results[job_id] = image_data
                except Exception as e:
                    print(f"保存图像失败: {e}")
                    self.draw_results[job_id] = None
            else:
                self._debug_log(f"任务执行失败: {job_id}, draw_success={draw_success}, has_image={image_data is not None}")
                self.draw_results[job_id] = None
            
            # 清理futures
            if job_id in self.active_draw_futures:
                del self.active_draw_futures[job_id]
                self._debug_log(f"清理任务 {job_id} 的futures")
            
            # 释放端口
            self._release_port(job_id)
            self._debug_log(f"任务 {job_id} 后台线程结束")
        
        # 启动后台线程同时发送请求并保存结果
        threading.Thread(target=_send_and_save_results, daemon=True).start()
        
        # 立即返回，不等待draw完成
        return True


class RealBackend(Backend):
    """真实后端 - 连接真实的后端节点"""
    
    def __init__(
        self,
        num_nodes: int,
        node_ips: List[str],
        runtime_config
    ):
        """初始化真实后端
        
        Args:
            num_nodes: 节点数量
            node_ips: 节点IP地址列表
            runtime_config: 运行时配置对象
        """
        self.num_nodes = num_nodes
        self.node_ips = node_ips
        self.runtime_config = runtime_config
        
        # 节点状态缓存
        self.node_model_ids: List[int] = [0] * num_nodes
        self.node_remaining_times: List[float] = [0.0] * num_nodes
        self.last_update_time: float = time.time()
        
        # 任务管理器
        self.task_manager = TaskManager(
            node_ips=node_ips,
            runtime_config=runtime_config,
            timeout=runtime_config.api_timeout
        )
    
    def reset(self) -> None:
        """重置后端状态"""
        # 销毁所有节点
        for i in range(self.num_nodes):
            self.task_manager.destroy_node(i)
        
        # 重置状态
        self.node_model_ids = [0] * self.num_nodes
        self.node_remaining_times = [0.0] * self.num_nodes
        self.last_update_time = time.time()
    
    def get_node_state(self, node_index: int) -> np.ndarray:
        """获取单个节点的状态"""
        if node_index < 0 or node_index >= self.num_nodes:
            return np.zeros(2, dtype=np.float32)
        
        # 从API获取最新的剩余时间
        remaining = self.task_manager.get_remaining_time(node_index)
        self.node_remaining_times[node_index] = remaining
        
        # 返回归一化状态
        return np.array([
            remaining / 30.0,  # 归一化剩余时间
            float(self.node_model_ids[node_index]) / 30.0  # 归一化模型ID
        ], dtype=np.float32)
    
    def get_all_states(self) -> np.ndarray:
        """获取所有节点的状态"""
        states = []
        for i in range(self.num_nodes):
            states.append(self.get_node_state(i))
        return np.stack(states, axis=0).astype(np.float32)
    
    def assign_task(self, task: SDTask) -> bool:
        """分配任务给节点（真实模式）"""
        # 注意：这个函数必须在env.py修改model_id之前调用
        # 这样才能正确计算affected_groups
        
        affected_groups = []
        if task.reload_required:
            # 如果需要reload，根据当前model_id状态计算affected_groups
            current_groups: Set[int] = set()
            for idx in task.assigned_node_indices:
                model_id = self.node_model_ids[idx]
                if model_id != 0:
                    current_groups.add(model_id)
            
            # 收集每个组的所有节点
            for group_id in current_groups:
                group_nodes = [
                    i for i in range(self.num_nodes)
                    if self.node_model_ids[i] == group_id
                ]
                if group_nodes:
                    affected_groups.append(group_nodes)
        
        # 执行任务（destroy → init → draw）
        success = self.task_manager.execute_task(task, affected_groups)
        
        if success:
            # 更新本地model_id状态（与真实后端同步）
            if task.reload_required:
                # Unload涉及的组
                for group_nodes in affected_groups:
                    for idx in group_nodes:
                        self.node_model_ids[idx] = 0
                
                # Load新组
                new_group_id = max(self.node_model_ids) + 1 if any(self.node_model_ids) else 1
                for idx in task.assigned_node_indices:
                    self.node_model_ids[idx] = new_group_id
            
            # 更新剩余时间估算
            for idx in task.assigned_node_indices:
                estimated_time = task.steps * 0.1
                self.node_remaining_times[idx] = estimated_time
        
        return success
    
    def update(self, dt: float, now: float) -> None:
        """更新后端状态
        
        在真实模式下，主要是同步剩余时间
        """
        # 更新所有节点的剩余时间（从API拉取）
        for i in range(self.num_nodes):
            remaining = self.task_manager.get_remaining_time(i)
            self.node_remaining_times[i] = remaining
        
        self.last_update_time = time.time()
    
    def is_node_idle(self, node_index: int) -> bool:
        """检查节点是否空闲"""
        if node_index < 0 or node_index >= self.num_nodes:
            return False
        
        # 从API获取最新状态
        remaining = self.task_manager.get_remaining_time(node_index)
        self.node_remaining_times[node_index] = remaining
        
        return remaining <= 0.0
    
    def get_node_model_id(self, node_index: int) -> int:
        """获取节点当前加载的模型ID"""
        if node_index < 0 or node_index >= self.num_nodes:
            return 0
        return self.node_model_ids[node_index]
    
    def set_node_model_id(self, node_index: int, model_id: int) -> None:
        """设置节点的模型ID"""
        if 0 <= node_index < self.num_nodes:
            self.node_model_ids[node_index] = model_id
    
    def get_node_remaining_time(self, node_index: int) -> float:
        """获取节点剩余执行时间"""
        if node_index < 0 or node_index >= self.num_nodes:
            return 0.0
        
        # 从API获取最新值
        remaining = self.task_manager.get_remaining_time(node_index)
        self.node_remaining_times[node_index] = remaining
        return remaining
    
    def _get_affected_groups(self, new_node_indices: List[int]) -> List[List[int]]:
        """获取需要卸载的进程组
        
        Args:
            new_node_indices: 新任务要使用的节点索引列表
            
        Returns:
            受影响的进程组列表（每个组是节点索引列表）
        """
        affected_groups: Set[int] = set()
        
        # 找出所有受影响的模型组
        for idx in new_node_indices:
            model_id = self.node_model_ids[idx]
            if model_id != 0:  # 0表示未加载模型
                affected_groups.add(model_id)
        
        # 收集每个组的所有节点
        groups_to_destroy: List[List[int]] = []
        for group_id in affected_groups:
            group_nodes = [
                i for i in range(self.num_nodes)
                if self.node_model_ids[i] == group_id
            ]
            if group_nodes:
                groups_to_destroy.append(group_nodes)
        
        return groups_to_destroy

