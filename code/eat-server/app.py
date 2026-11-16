import os
import tempfile
import threading
import time
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from torch import distributed as dist
from starlette.background import BackgroundTask

from distrifuser.pipelines import DistriSDPipeline
from distrifuser.utils import DistriConfig, is_power_of_2

import argparse 
import uvicorn

os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"

app = FastAPI()
_lock = threading.Lock()

def setup_gpu_device(device_id: int):
    """根据传入的设备ID，设置CUDA_VISIBLE_DEVICES环境变量"""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        print(f"Success: CUDA_VISIBLE_DEVICES has been set to '{device_id}'.")
        print("This process will now only see this GPU as 'cuda:0'.")
    except Exception as e:
        print(f"Error setting CUDA_VISIBLE_DEVICES: {e}")
        raise e


BASE_PIPE: Optional[DistriSDPipeline] = None
CURRENT_JOB_ID: Optional[str] = None


BASE_STEP_TIME: float = 0.1  
TOTAL_STEPS: int = 0  
CURRENT_STEP: int = 0  
STEP_START_TIME: float = 0.0  
STEP_TIMES: list = []  
ESTIMATED_REMAIN_TIME: float = 0.0  

class InitReq(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    master_addr: str = Field(..., description="Rendezvous master address (IP or hostname)")
    master_port: int = Field(..., description="Rendezvous master port")
    world_size: int = Field(..., description="Total number of processes across all nodes")
    rank: int = Field(..., description="Global rank for this process")
    local_rank: int = Field(..., description="Local rank (GPU index) on this node")

    
    height: int = 1024
    width: int = 1024
    warmup_steps: int = 4
    mode: str = "corrected_async_gn"  
    parallelism: str = "patch"  
    split_batch: bool = True
    use_cuda_graph: bool = True
    guidance_scale: float = 5.0  


class DrawReq(BaseModel):
    job_id: str
    prompt: str
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    seed: int = 1234

@app.on_event("startup")
def _startup():
    global BASE_PIPE
    
    device_to_use = torch.device("cuda:0")
   
    torch.cuda.set_device(device_to_use)

    base_cfg = DistriConfig(
        height=1024, 
        width=1024, 
        warmup_steps=4, 
        init_pg=False,
        device=device_to_use  
    )
    
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:20171"
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:20171"
    BASE_PIPE = DistriSDPipeline.from_pretrained(
        distri_config=base_cfg,
        pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    del os.environ['HTTPS_PROXY']
    del os.environ['HTTP_PROXY']

@app.post("/destroy")
def destroy_pg():
    """Explicitly destroy the current process group if it exists."""
    global CURRENT_JOB_ID, _SHUTTING_DOWN, BASE_PIPE
    if not dist.is_initialized():
        print("not initialized, skip destroy")
        return {"status": "not initialized"}

    BASE_PIPE.teardown()
    
    
    with _lock:
        try:
            dist.destroy_process_group()
            torch.cuda.synchronize()

            CURRENT_JOB_ID = None
            return {"status": "destroyed"}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"destroy failed: {e}"})

@app.post("/init")
def init_pg(req: InitReq):
    """Initialize/reinitialize the torch.distributed process group and reconfigure pipeline.

    Returns JSON with rank/world_size and status.
    """
    
    global CURRENT_JOB_ID, BASE_PIPE

    if BASE_PIPE is None:
        raise HTTPException(status_code=500, detail="Base pipeline is not loaded")

    if not is_power_of_2(req.world_size):
        return JSONResponse(status_code=400, content={"error": "world_size must be a power of 2"})

    
    do_cfg = req.guidance_scale > 1.0

    
    if dist.is_initialized():
        CURRENT_JOB_ID = req.job_id
        return {"status": "already initialized","job_id": CURRENT_JOB_ID}
        

    
    device_to_use = torch.device("cuda:0")
    torch.cuda.set_device(device_to_use)
    init_method = f"tcp://{req.master_addr}:{req.master_port}"
    
    try:
        kwargs = dict(
            backend="nccl",
            init_method=init_method,
            rank=req.rank,
            world_size=req.world_size
        )
        kwargs["device_id"] = torch.cuda.current_device()
        
        dist.init_process_group(**kwargs)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"init_process_group failed: {e}"})

    with _lock:

        
        cfg = DistriConfig(
            height=req.height,
            width=req.width,
            do_classifier_free_guidance=do_cfg,
            split_batch=req.split_batch,
            warmup_steps=req.warmup_steps,
            mode=req.mode,
            use_cuda_graph=req.use_cuda_graph,
            parallelism=req.parallelism,
            init_pg=False,
            device=device_to_use,
        )

        try:
            BASE_PIPE.reconfigure(cfg)
        except Exception as e:
            
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:
                pass
            return JSONResponse(status_code=500, content={"error": f"pipeline reconfigure failed: {e}"})

    CURRENT_JOB_ID = req.job_id

    try:
        dist.barrier()
    except Exception:
        
        return JSONResponse(status_code=500, content={"error": "barrier failed after init"})

    return {"status": "ok", "job_id": CURRENT_JOB_ID, "rank": cfg.rank, "world_size": cfg.world_size}


def step_callback(pipe, step_index, timestep, callback_kwargs):
    """Callback function to track step execution time"""
    global CURRENT_STEP, STEP_START_TIME, STEP_TIMES, ESTIMATED_REMAIN_TIME, TOTAL_STEPS
    
    current_time = time.time()
    
    if CURRENT_STEP > 0:
        
        step_time = current_time - STEP_START_TIME
        STEP_TIMES.append(step_time)
        
        
        avg_step_time = sum(STEP_TIMES) / len(STEP_TIMES)
        remaining_steps = TOTAL_STEPS - CURRENT_STEP
        ESTIMATED_REMAIN_TIME = avg_step_time * remaining_steps
    
    CURRENT_STEP = step_index + 1
    STEP_START_TIME = current_time
    
    return callback_kwargs

@app.post("/draw")
def draw(req: DrawReq):
    print("get draw request")
    """Run inference using current PG and config. Returns FileResponse (PNG) on rank 0, JSON otherwise."""
    global TOTAL_STEPS, CURRENT_STEP, STEP_TIMES, ESTIMATED_REMAIN_TIME, BASE_STEP_TIME, STEP_START_TIME
    
    if not dist.is_initialized():
        print("process group not initialized; call /init first")
        return JSONResponse(status_code=400, content={"error": "process group not initialized; call /init first"})
    if req.job_id != CURRENT_JOB_ID:
        print("job_id mismatch with current process group")
        return JSONResponse(status_code=409, content={"error": "job_id mismatch with current process group"})

    pipe = BASE_PIPE
    cfg = pipe.distri_config

    
    requested_do_cfg = req.guidance_scale > 1.0
    if requested_do_cfg != cfg.do_classifier_free_guidance:
        print("guidance_scale incompatible with current config; re-init required")
        return JSONResponse(status_code=400, content={"error": "guidance_scale incompatible with current config; re-init required"})

    with _lock:
        
        TOTAL_STEPS = req.num_inference_steps
        CURRENT_STEP = 0
        STEP_TIMES = []
        ESTIMATED_REMAIN_TIME = TOTAL_STEPS * BASE_STEP_TIME
        STEP_START_TIME = time.time()
        
        try:
            pipe.set_progress_bar_config(disable=cfg.rank != 0)
            image = pipe(
                prompt=req.prompt,
                generator=torch.Generator(device="cuda").manual_seed(req.seed),
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
                callback_on_step_end=step_callback,
            ).images[0]
            torch.cuda.synchronize()
            
            
            ESTIMATED_REMAIN_TIME = 0.0
        except Exception as e:
            print(f"inference failed: {e}")
            return JSONResponse(status_code=500, content={"error": f"inference failed: {e}"})

        try:
            print("barrier")
            dist.barrier()
            print("barrier done")
        except Exception:
            
            print("barrier failed")
            pass

        if cfg.rank == 0:
            
            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            try:
                image.save(path)
            except Exception as e:
                
                try:
                    os.remove(path)
                except Exception:
                    pass
                return JSONResponse(status_code=500, content={"error": f"saving image failed: {e}"})

            
            return FileResponse(path, media_type="image/png", filename="result.png", background=BackgroundTask(os.remove, path))
        else:
            return {"status": "ok", "rank": cfg.rank}

@app.get("/remain_time")
def remain_time():
    return {"remain_time": ESTIMATED_REMAIN_TIME}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DistriFuser FastAPI Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--device-id", type=int, default=0, help="The physical ID of the GPU to use")
    
    args = parser.parse_args()

    
    print(f"Preparing to start server on port {args.port} using physical GPU {args.device_id}...")
    setup_gpu_device(args.device_id)

    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
