"""Curate the raw extraction into a small, story-ready dataset.

Input : tmp/extracted  (every tag, every point; ~125 MB)
Output: tmp/curated    (curated tags, <= `--max-points` points each; a few MB)

Also writes two aggregate tables used by the chapter documents:

    runs_index.csv     one row per run: campaign, variant, algorithm, run, steps, rewards
    by_algorithm.csv   one row per (campaign, variant, algorithm): mean/std over seeds

Curated tags keep the learning signal (reward, length, quality, latency, eta, sigma,
losses, action statistics, epochs) and drop the per-parameter norm/sparsity dumps
that made EAT runs ~1 GB each.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

DROP_CONTAINS = ("_norm/", "_sparsity/", "norm/", "sparsity/")
KEEP_PREFIXES = ("test/", "train/", "update/", "save/")

# Run-directory names are `YYYY-mm-dd_HH-MM-SS`.
RUN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
DENSITY_RE = re.compile(r"^\d+_\d\.\d+$")
CAMPAIGN_BASE = {
    "fig6-main": "EAT基线对比/A",
    "trials": "EAT基线对比/ATry",
    "ablation": "EAT基线对比/B",
    "learning-rate": "EAT基线对比/DatA",
    "fallback-rerun": "EAT基线对比/fallback",
    "misc": "EAT基线对比/2026-08-12_19-46-31",
    "scale-density": "EAT基线对比/zzz",
    "scale-density-z": "z/A/EAT-Train/EAT-Train/runs",
    "model-copy": "ModelCopy",
}


def keep_tag(tag: str) -> bool:
    if any(bad in tag for bad in DROP_CONTAINS):
        return False
    return tag.startswith(KEEP_PREFIXES)


def downsample(points: list[tuple[int, float]], max_points: int) -> list[tuple[int, float]]:
    n = len(points)
    if n <= max_points:
        return points
    stride = math.ceil(n / max_points)
    keep = list(range(0, n, stride))
    if keep[-1] != n - 1:
        keep.append(n - 1)
    return [points[i] for i in keep]


def fmt(value: float) -> str:
    if value != value:  # NaN
        return "nan"
    if abs(value) >= 1e6 or (value != 0 and abs(value) < 1e-4):
        return f"{value:.6e}"
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def classify(source_file: Path, campaign: str) -> tuple[str, str]:
    """Return (variant, algorithm) from the run's directory chain."""
    base = CAMPAIGN_BASE.get(campaign, "")
    try:
        rel = source_file.relative_to(Path(base).anchor or "/")
    except Exception:  # noqa: BLE001
        rel = Path(source_file)
    parts = list(Path(source_file).parts)
    idx = None
    for i, part in enumerate(parts):
        if RUN_RE.match(part):
            idx = i
    if idx is None:
        return "", "unknown"
    chain = [p for p in parts[: idx] if not RUN_RE.match(p)]
    algorithm = chain[-1] if chain else "unknown"
    variant = ""
    if len(chain) >= 2 and DENSITY_RE.match(chain[-2]):
        variant = chain[-2]
    if algorithm.endswith("3e") or algorithm in {"3e-4"}:
        for candidate in reversed(chain[:-1]):
            if candidate and not DENSITY_RE.match(candidate) and candidate != "3e-4":
                algorithm = candidate
                break
    if DENSITY_RE.match(algorithm) and len(chain) >= 2:
        variant = algorithm
        algorithm = chain[-2] if len(chain) >= 2 else "unknown"
    return variant, algorithm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-points", type=int, default=600)
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict] = []
    for run_json in sorted(in_root.rglob("run.json")):
        meta = json.loads(run_json.read_text(encoding="utf-8"))
        campaign = meta["campaign"]
        variant, algorithm = classify(Path(meta["source_file"]), campaign)
        run_dir = run_json.parent

        tags: dict[str, list[tuple[int, float]]] = {}
        scalars_csv = run_dir / "scalars.csv"
        with scalars_csv.open("r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) != 3:
                    continue
                tag, step, value = row
                if not keep_tag(tag):
                    continue
                try:
                    tags.setdefault(tag, []).append((int(step), float(value)))
                except ValueError:
                    continue

        target = out / campaign / (variant or "_") / algorithm / meta["run_id"]
        target.mkdir(parents=True, exist_ok=True)
        lines = ["tag,step,value"]
        kept = {}
        for tag, points in sorted(tags.items()):
            points.sort()
            pts = downsample(points, args.max_points)
            kept[tag] = len(pts)
            for step, value in pts:
                lines.append(f"{tag},{step},{fmt(value)}")
        (target / "curves.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

        test = meta["summary"].get("test/reward", {})
        train = meta["summary"].get("train/reward", {})
        eta = meta["summary"].get("train/ddim_eta", {})
        (target / "run.json").write_text(
            json.dumps(
                {
                    "campaign": campaign,
                    "variant": variant,
                    "algorithm": algorithm,
                    "run_id": meta["run_id"],
                    "env_steps": test.get("step_max", 0),
                    "test_reward": {
                        "first": test.get("first"),
                        "last": test.get("last"),
                        "best": test.get("best"),
                        "best_step": test.get("best_step"),
                        "n": test.get("n"),
                    },
                    "train_reward_last": train.get("last"),
                    "ddim_eta_last": eta.get("last"),
                    "curated_tags": sorted(kept),
                    "source_file": meta["source_file"],
                    "source_mb": round(meta["source_bytes"] / 1048576, 2),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        run_rows.append(
            {
                "campaign": campaign,
                "variant": variant,
                "algorithm": algorithm,
                "run_id": meta["run_id"],
                "env_steps": test.get("step_max", 0),
                "test_reward_first": test.get("first"),
                "test_reward_last": test.get("last"),
                "test_reward_best": test.get("best"),
                "test_reward_best_step": test.get("best_step"),
                "n_eval_points": test.get("n", 0),
                "ddim_eta_last": eta.get("last"),
                "source_mb": round(meta["source_bytes"] / 1048576, 2),
            }
        )

    run_rows.sort(key=lambda r: (r["campaign"], r["variant"], r["algorithm"], r["run_id"]))
    cols = list(run_rows[0]) if run_rows else []
    with (out / "runs_index.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in run_rows:
            writer.writerow(row)

    # per (campaign, variant, algorithm) aggregation over seeds
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in run_rows:
        groups.setdefault((row["campaign"], row["variant"], row["algorithm"]), []).append(row)
    agg_rows = []
    for (campaign, variant, algorithm), rows in sorted(groups.items()):
        lasts = [r["test_reward_last"] for r in rows if r["test_reward_last"] is not None]
        bests = [r["test_reward_best"] for r in rows if r["test_reward_best"] is not None]
        steps = [r["env_steps"] for r in rows if r["env_steps"]]
        agg_rows.append(
            {
                "campaign": campaign,
                "variant": variant,
                "algorithm": algorithm,
                "n_runs": len(rows),
                "env_steps_max": max(steps) if steps else 0,
                "final_mean": round(statistics.fmean(lasts), 3) if lasts else None,
                "final_std": round(statistics.stdev(lasts), 3) if len(lasts) > 1 else 0.0,
                "best_mean": round(statistics.fmean(bests), 3) if bests else None,
                "best_std": round(statistics.stdev(bests), 3) if len(bests) > 1 else 0.0,
                "per_seed_final": "|".join(f"{v:.2f}" for v in lasts),
            }
        )
    with (out / "by_algorithm.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(agg_rows[0]))
        writer.writeheader()
        for row in agg_rows:
            writer.writerow(row)

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"runs={len(run_rows)} groups={len(agg_rows)} size={total/1048576:.1f} MB -> {out}")
    for row in agg_rows:
        if row["campaign"] in {"fig6-main", "ablation", "learning-rate", "fallback-rerun"}:
            print(
                f"  {row['campaign']:16s} {row['variant']:10s} {row['algorithm']:12s} "
                f"n={row['n_runs']} final={row['final_mean']}±{row['final_std']} "
                f"best={row['best_mean']} steps={row['env_steps_max']:,}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
