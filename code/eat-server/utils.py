
from typing import Dict, Any

import torch
from torch import distributed as dist
from torch.distributed.distributed_c10d import GroupMember


def _safe_getattr(obj, name, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default

def _stream_info(stream) -> Dict[str, Any]:
    info = {"id": hex(id(stream))}
    try:
        
        info["idle"] = bool(stream.query())
    except Exception:
        info["idle"] = None
    try:
        info["priority"] = stream.priority
    except Exception:
        pass
    return info

def inspect_pipeline_state(base_pipe) -> Dict[str, Any]:
    if base_pipe is None:
        return {"present": False}

    d: Dict[str, Any] = {"present": True}
    cfg = base_pipe.distri_config
    d["device"] = str(cfg.device)
    d["height"], d["width"] = cfg.height, cfg.width
    d["use_cuda_graph"] = bool(_safe_getattr(cfg, "use_cuda_graph", False))
    d["parallelism"] = _safe_getattr(cfg, "parallelism", None)
    d["do_cfg"] = bool(_safe_getattr(cfg, "do_classifier_free_guidance", False))

    unet = base_pipe.pipeline.unet
    d["unet_cls"] = unet.__class__.__name__

    
    graph_handles = []
    for name in ("_cuda_graphs", "cuda_graphs"):
        gs = _safe_getattr(unet, name, None)
        if gs:
            try:
                graph_handles.extend([hex(id(g)) for g in (gs if isinstance(gs, (list, tuple)) else [gs])])
            except Exception:
                graph_handles.append(f"<{name}-exists>")
    d["cuda_graph_active"] = len(graph_handles) > 0
    d["cuda_graph_handles"] = graph_handles

    
    d["has_static_outputs"] = _safe_getattr(unet, "_static_outputs", None) is not None
    d["has_static_inputs"]  = base_pipe.static_inputs is not None

    
    cm = _safe_getattr(unet, "comm_manager", None)
    d["has_comm_manager"] = cm is not None
    if cm is not None:
        d["comm_manager_cls"] = cm.__class__.__name__
        for k in ("numel", "world_size", "group_size", "n_device_per_batch"):
            v = _safe_getattr(cm, k, None)
            if v is not None: d[f"cm_{k}"] = v
        
        pg_attrs = []
        for name in ("pg", "process_group", "pgs", "subgroups", "_pgs", "_subgroups"):
            v = _safe_getattr(cm, name, None)
            if v:
                try:
                    items = v if isinstance(v, (list, tuple)) else [v]
                    pg_attrs.append({ "attr": name, "count": len(items), "ids": [hex(id(x)) for x in items if x is not GroupMember.NON_GROUP_MEMBER] })
                except Exception:
                    pg_attrs.append({ "attr": name, "present": True })
        if pg_attrs: d["comm_manager_groups"] = pg_attrs

    
    streams = []
    for name in ("streams", "comm_streams", "_streams"):
        v = _safe_getattr(unet, name, None)
        if v:
            items = v if isinstance(v, (list, tuple)) else [v]
            streams.append({
                "attr": name,
                "count": len(items),
                "list": [_stream_info(s) for s in items if s is not None],
            })
    d["streams"] = streams

    
    try:
        d["cuda_current_device"] = int(torch.cuda.current_device())
        d["cuda_memory_allocated_MB"] = round(torch.cuda.memory_allocated() / 1024**2, 2)
        d["cuda_memory_reserved_MB"]  = round(torch.cuda.memory_reserved() / 1024**2, 2)
        d["cuda_max_memory_reserved_MB"] = round(torch.cuda.max_memory_reserved() / 1024**2, 2)
    except Exception:
        pass

    
    try:
        d["dist_backend"] = dist.get_backend()
        d["dist_rank"]    = dist.get_rank()
        d["dist_world_size"] = dist.get_world_size()
    except Exception:
        pass

    return d
