# runtime_config_utils.py

import json
from typing import List, Dict, Any, Optional

# 假设您真正的运行时配置函数位于一个名为 "config" 的模块中
# 为了避免命名冲突，我们导入时使用别名
from config import set_runtime_config as apply_real_config
from config import set_global_config

# 定义一个常量，用于指定配置文件的标准名称
CONFIG_FILE_PATH = "runtime_config.json"

def create_runtime_config(
    num_nodes: int, 
    node_hostnames: List[str],
    service_port: int = 8000,
    master_port: int = 29500
) -> Dict[str, Any]:
    """
    根据节点信息，生成一个标准的运行时配置字典。
    这个函数是纯粹的，不依赖任何环境变量。
    """
    # 假设主控节点是列表中的第一个节点
    master_addr = node_hostnames[0] if node_hostnames else "127.0.0.1"
    
    # 根据主机名和端口生成服务节点的IP:PORT列表
    node_ips = [f"{hostname}:{service_port}" for hostname in node_hostnames]
    
    # 构建与您原始代码结构相同的配置字典
    config_data = {
        "mode": "real",
        "node_ips": node_ips,
        "master_addr": master_addr,
        "master_port": master_port,
        "output_dir": "outputs_real",
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 50,
        "api_timeout": 60.0,
        "debug": True,
    }
    
    # 将全局配置也纳入管理，虽然它不属于runtime_config
    # 但在这里一起生成，可以统一管理所有动态配置
    global_config_data = {
        "node_num": num_nodes,
        "max_visible_tasks": 8,
    }

    return {
        "global_config": global_config_data,
        "runtime_config": config_data
    }


def set_runtime_config(config_path: Optional[str] = None) -> None:
    """
    从JSON文件中读取配置，并调用真正的配置函数来应用它。
    这是您将在主程序中调用的唯一函数。
    """
    if config_path is None:
        config_path = CONFIG_FILE_PATH

    print(f"Reading configuration from '{config_path}'...")
    
    try:
        with open(config_path, 'r') as f:
            full_config = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Configuration file '{config_path}' not found!")
        print("Please ensure you run 'init_runtime_config.py' before your main experiment script.")
        exit(1) # 严重错误，直接退出
    
    global_cfg = full_config.get("global_config")
    runtime_cfg = full_config.get("runtime_config")

    if not global_cfg or not runtime_cfg:
        print(f"ERROR: Configuration file '{config_path}' is malformed.")
        exit(1)

    # 调用您项目中的真实配置函数，并将加载的配置作为参数传入
    # **runtime_cfg 会将字典解包为关键字参数
    set_global_config(**global_cfg)
    apply_real_config(**runtime_cfg)
    
    print("Runtime environment configured successfully.")

