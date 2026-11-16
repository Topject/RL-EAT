"""Read a torch .pth checkpoint's parameter keys, shapes and parameter counts
without installing torch.

torch.save writes a zip whose `*/data.pkl` is a pickle. We walk it with
`pickletools.genops` and pair every BINUNICODE key with the following
BINPERSISTENT storage descriptor, which carries `numel`. That yields an exact
parameter inventory: names, element counts, per-submodule totals.

Usage: python ckpt_inventory.py <out_dir> <run_label>=<file.pth> [...]
"""
from __future__ import annotations

import json
import pickletools
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

KEY_RE = re.compile(r"^[A-Za-z0-9_.]+\.(weight|bias|betas|alphas_cumprod|eta_param)$")
STRING_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "SHORT_BINSTRING"}


def inventory(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        pkl_names = [n for n in zf.namelist() if n.endswith("data.pkl")]
        best = None
        for name in pkl_names:
            data = zf.read(name)
            if b"weight" in data:
                best = data
                break
        if best is None:
            raise RuntimeError(f"no tensor pickle in {path}")

    keys: list[str] = []
    pending: str | None = None
    last_int: int | None = None
    params: list[tuple[str, int]] = []
    int_ops = {"BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4", "INT"}
    for op, arg, _pos in pickletools.genops(best):
        opname = op.name
        if opname in int_ops and isinstance(arg, int):
            last_int = arg
        elif opname in STRING_OPS and isinstance(arg, str):
            if KEY_RE.match(arg):
                pending = arg
                keys.append(arg)
                last_int = None
        elif opname in {"BINPERSID", "PERSID"}:
            if pending is not None and last_int is not None:
                params.append((pending, int(last_int)))
            pending = None
            last_int = None

    def leaf(key: str) -> str:
        parts = key.split(".")
        return ".".join(parts[:-1]) if len(parts) > 1 else key

    by_leaf: dict[str, int] = defaultdict(int)
    by_root: dict[str, int] = defaultdict(int)
    for key, numel in params:
        by_leaf[leaf(key)] += numel
        by_root[key.split(".")[0]] += numel

    all_keys = [k for k, _ in params]
    joined = "\n".join(all_keys)
    flags = {
        "learnable_eta": ("eta_param" in joined),
        "ddim_marker": any(k.endswith("eta_param") for k in all_keys),
        "diffusion": any(k.endswith("betas") or "alphas_cumprod" in k for k in all_keys),
        "cross_attention": bool(re.search(r"cross", joined, re.I)),
        "self_attention": bool(re.search(r"self_attn|selfattn", joined, re.I)),
        "attention_any": bool(re.search(r"attn|attention", joined, re.I)),
    }
    return {
        "file": str(path),
        "size_mb": round(path.stat().st_size / 1048576, 3),
        "n_params_tensors": len(params),
        "total_numel": sum(n for _, n in params),
        "by_root": dict(sorted(by_root.items(), key=lambda kv: -kv[1])),
        "by_leaf": dict(sorted(by_leaf.items(), key=lambda kv: -kv[1])),
        "flags": flags,
        "keys": all_keys,
    }


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for spec in sys.argv[2:]:
        label, _, raw = spec.partition("=")
        info = inventory(Path(raw))
        report[label] = info
        print(
            f"{label:28s} params={info['total_numel']:>10,d} tensors={info['n_params_tensors']:>4d} "
            f"diffusion={info['flags']['diffusion']} eta={info['flags']['learnable_eta']} "
            f"attn={info['flags']['attention_any']}"
        )
        roots = ", ".join(f"{k}:{v:,}" for k, v in info["by_root"].items())
        print(f"{'':28s} roots -> {roots}")
    (out / "ckpt_inventory.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
