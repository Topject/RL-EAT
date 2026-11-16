import io
import os
import tempfile
import threading
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from torch import distributed as dist
from starlette.background import BackgroundTask

from distrifuser.pipelines import DistriSDXLPipeline
from distrifuser.utils import DistriConfig, is_power_of_2


app = FastAPI()
_lock = threading.Lock()

# Global state per process
BASE_PIPE: Optional[DistriSDXLPipeline] = None
CURRENT_JOB_ID: Optional[str] = None


class InitReq(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    master_addr: str = Field(..., description="Rendezvous master address (IP or hostname)")
    master_port: int = Field(..., description="Rendezvous master port")
    world_size: int = Field(..., description="Total number of processes across all nodes")
    rank: int = Field(..., description="Global rank for this process")
    local_rank: int = Field(..., description="Local rank (GPU index) on this node")

    # DistriFuser / pipeline config
    height: int = 1024
    width: int = 1024
    warmup_steps: int = 4
    mode: str = "corrected_async_gn"  # ["separate_gn","stale_gn","corrected_async_gn","sync_gn","full_sync","no_sync"]
    parallelism: str = "patch"  # ["patch","tensor","naive_patch"]
    split_batch: bool = True
    use_cuda_graph: bool = True
    guidance_scale: float = 5.0  # used to set do_classifier_free_guidance and groups


class DrawReq(BaseModel):
    job_id: str
    prompt: str
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    seed: int = 1234


@app.on_event("startup")
def _startup():
    global BASE_PIPE
    # Select default device 0 at startup; it can be moved later during reconfigure
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))

    # Load once without initializing a process group to keep weights resident
    base_cfg = DistriConfig(init_pg=False)
    BASE_PIPE = DistriSDXLPipeline.from_pretrained(
        distri_config=base_cfg,
        pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
        variant="fp16",
        use_safetensors=True,
    )


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

    # do_cfg determines subgroup topology; must be consistent during a job
    do_cfg = req.guidance_scale > 1.0

    with _lock:
        # Destroy any existing PG first
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
                torch.cuda.synchronize()
            except Exception:
                pass

        # Bind to the requested local GPU
        torch.cuda.set_device(req.local_rank)

        # Initialize PG via explicit init_method/rank/world_size
        init_method = f"tcp://{req.master_addr}:{req.master_port}"
        try:
            dist.init_process_group(
                backend="nccl",
                init_method=init_method,
                rank=req.rank,
                world_size=req.world_size,
            )
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"init_process_group failed: {e}"})

        # Build DistriConfig without re-initializing PG; fix device by local_rank
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
            device=torch.device(f"cuda:{req.local_rank}"),
        )

        try:
            BASE_PIPE.reconfigure(cfg)
        except Exception as e:
            # Clean up PG on failure
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
            # If barrier fails, surface but keep process alive for further attempts
            return JSONResponse(status_code=500, content={"error": "barrier failed after init"})

        return {"status": "ok", "job_id": CURRENT_JOB_ID, "rank": cfg.rank, "world_size": cfg.world_size}


@app.post("/draw")
def draw(req: DrawReq):
    """Run inference using current PG and config. Returns FileResponse (PNG) on rank 0, JSON otherwise."""
    if not dist.is_initialized():
        return JSONResponse(status_code=400, content={"error": "process group not initialized; call /init first"})
    if req.job_id != CURRENT_JOB_ID:
        return JSONResponse(status_code=409, content={"error": "job_id mismatch with current process group"})

    pipe = BASE_PIPE
    cfg = pipe.distri_config

    # Ensure requested guidance_scale matches current config topology (do_cfg)
    requested_do_cfg = req.guidance_scale > 1.0
    if requested_do_cfg != cfg.do_classifier_free_guidance:
        return JSONResponse(status_code=400, content={"error": "guidance_scale incompatible with current config; re-init required"})

    with _lock:
        try:
            pipe.set_progress_bar_config(disable=cfg.rank != 0)
            image = pipe(
                prompt=req.prompt,
                generator=torch.Generator(device="cuda").manual_seed(req.seed),
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
            ).images[0]
            torch.cuda.synchronize()
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"inference failed: {e}"})

        try:
            dist.barrier()
        except Exception:
            # proceed for rank 0 to still return the image if possible
            pass

        if cfg.rank == 0:
            # Save to a temp file and return as FileResponse
            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            try:
                image.save(path)
            except Exception as e:
                # Best-effort cleanup
                try:
                    os.remove(path)
                except Exception:
                    pass
                return JSONResponse(status_code=500, content={"error": f"saving image failed: {e}"})

            # Starlette will stream this file; the caller can download or preview
            return FileResponse(path, media_type="image/png", filename="result.png", background=BackgroundTask(os.remove, path))
        else:
            return {"status": "ok", "rank": cfg.rank}


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()

    # Bind to 0.0.0.0 for Docker/container use
    uvicorn.run(app, host="0.0.0.0", port=args.port)


