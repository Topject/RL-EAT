"""Extract TensorBoard scalar curves from every EAT reproduction run.

Walks the author's log tree (`EATlogs/EAT基线对比`, `EATlogs/z`, `EATlogs/ModelCopy`),
parses each `events.out.tfevents.*` file with the dependency-free reader in
`safe_tfreader.py`, and writes, per run:

    <out>/<campaign>/<algorithm>/<run_id>/scalars.csv   tag,step,value (downsampled)
    <out>/<campaign>/<algorithm>/<run_id>/run.json      metadata + summary stats

plus one flat index:

    <out>/runs_index.csv
    <out>/runs_index.json

Downsampling keeps at most `--max-points` points per tag by a uniform stride and
always keeps the first and last point, so curves stay faithful.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

from safe_tfreader import read_scalars

# (campaign label, relative path under the log root, human note)
CAMPAIGNS = [
    ("fig6-main", "EAT基线对比/A", "Figure 6 main comparison: 7 algorithms x 3 seeds"),
    ("trials", "EAT基线对比/ATry", "early exploratory runs before the naming settled"),
    ("ablation", "EAT基线对比/B", "SAC-CA / D2SAC-SA / EAT-CA ablations"),
    ("learning-rate", "EAT基线对比/DatA", "learning-rate 3e-4 arms"),
    ("fallback-rerun", "EAT基线对比/fallback", "second full sweep with the fallback env fix"),
    ("misc", "EAT基线对比/2026-08-12_19-46-31", "single leftover run"),
    ("scale-density", "EAT基线对比/zzz", "node-count / density runs"),
    ("scale-density-z", "z/A/EAT-Train/EAT-Train/runs", "earlier scale / density snapshot"),
    ("model-copy", "ModelCopy", "the single model kept as a reference copy"),
]

STEP_RE = re.compile(r"events\.out\.tfevents\.(\d+)\.")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-points", type=int, default=1200)
    ap.add_argument("--max-bytes", type=int, default=0, help="0 = no cap")
    return ap.parse_args()


def downsample(step: list[int], value: list[float], max_points: int) -> list[tuple[int, float]]:
    n = len(step)
    if n <= max_points:
        return list(zip(step, value))
    stride = math.ceil(n / max_points)
    keep = list(range(0, n, stride))
    if keep[-1] != n - 1:
        keep.append(n - 1)
    return [(step[i], value[i]) for i in keep]


def series_stats(points: list[tuple[int, float]]) -> dict:
    if not points:
        return {}
    steps = [s for s, _ in points]
    vals = [v for _, v in points]
    return {
        "n": len(points),
        "step_min": min(steps),
        "step_max": max(steps),
        "first": vals[0],
        "last": vals[-1],
        "best": max(vals),
        "best_step": steps[vals.index(max(vals))],
        "mean": sum(vals) / len(vals),
        "min": min(vals),
        "max": max(vals),
    }


def main() -> int:
    args = parse_args()
    log_root = Path(args.log_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for campaign, rel, note in CAMPAIGNS:
        base = log_root / rel
        if not base.exists():
            print(f"skip missing {rel}")
            continue
        for event_file in sorted(base.rglob("events.out.tfevents.*")):
            if event_file.stat().st_size == 0:
                continue
            rel_parts = event_file.relative_to(base).parts
            # algorithm = directory right above the timestamped run directory
            run_id = event_file.parent.name
            algorithm = rel_parts[-3] if len(rel_parts) >= 3 else "unknown"
            if algorithm == run_id:
                algorithm = "unknown"
            started = time.time()
            try:
                scalars = read_scalars(
                    event_file, limit_bytes=args.max_bytes or None
                )
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {event_file}: {exc}")
                continue
            if not scalars:
                continue
            target = out / campaign / algorithm / run_id
            target.mkdir(parents=True, exist_ok=True)

            summary = {}
            lines = ["tag,step,value"]
            for tag, points in sorted(scalars.items()):
                points = sorted(points, key=lambda p: p[0])
                summary[tag] = series_stats(points)
                for step, value in downsample([p[0] for p in points], [p[1] for p in points], args.max_points):
                    lines.append(f"{tag},{step},{value!r}")
            (target / "scalars.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

            m = STEP_RE.search(event_file.name)
            meta = {
                "campaign": campaign,
                "campaign_note": note,
                "algorithm": algorithm,
                "run_id": run_id,
                "source_file": str(event_file),
                "source_bytes": event_file.stat().st_size,
                "file_unix_time": int(m.group(1)) if m else None,
                "tags": sorted(scalars),
                "summary": summary,
                "parse_seconds": round(time.time() - started, 2),
            }
            (target / "run.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            test = summary.get("test/reward", {})
            train = summary.get("train/reward", {})
            rows.append(
                {
                    "campaign": campaign,
                    "algorithm": algorithm,
                    "run_id": run_id,
                    "source_mb": round(event_file.stat().st_size / 1048576, 2),
                    "env_steps": test.get("step_max", 0),
                    "test_reward_first": test.get("first"),
                    "test_reward_last": test.get("last"),
                    "test_reward_best": test.get("best"),
                    "test_reward_best_step": test.get("best_step"),
                    "test_reward_mean_of_run": round(test["mean"], 3) if test else None,
                    "train_reward_last": train.get("last"),
                    "n_test_points": test.get("n"),
                    "tags": len(scalars),
                }
            )
            print(
                f"{campaign:16s} {algorithm:9s} {run_id:20s} env={test.get('step_max', 0):>9,} "
                f"last={test.get('last', float('nan')):7.2f} best={test.get('best', float('nan')):7.2f} "
                f"points={test.get('n', 0):4d} ({meta['parse_seconds']}s)"
            )

    rows.sort(key=lambda r: (r["campaign"], r["algorithm"], r["run_id"]))
    cols = list(rows[0]) if rows else []
    csv_lines = [",".join(cols)]
    for row in rows:
        csv_lines.append(",".join("" if row[c] is None else str(row[c]) for c in cols))
    (out / "runs_index.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    (out / "runs_index.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nruns: {len(rows)} -> {out / 'runs_index.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
