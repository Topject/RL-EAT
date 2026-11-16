"""Flatten `data/<campaign>/<variant>/<algorithm>/<run>` into
`data/<campaign>/<algorithm>/<variant>__<run>` and add a `path` column to the
index tables, so every document can link a run directly.

Safe to re-run: already-flat directories are left alone.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data"
RENAME = {"_": ""}


def main() -> int:
    # 1. flatten
    for campaign in [p for p in ROOT.iterdir() if p.is_dir()]:
        for variant_dir in list(campaign.iterdir()):
            if not variant_dir.is_dir():
                continue
            children = list(variant_dir.iterdir())
            if not children:
                variant_dir.rmdir()
                continue
            # a variant dir contains algorithm dirs, which contain run dirs
            looks_like_variant = all(
                c.is_dir() and any(g.is_dir() for g in c.iterdir()) for c in children
            )
            if not looks_like_variant:
                continue
            variant = "" if variant_dir.name == "_" else variant_dir.name
            for algo_dir in children:
                target_algo = campaign / algo_dir.name
                target_algo.mkdir(exist_ok=True)
                for run_dir in list(algo_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    name = f"{variant}__{run_dir.name}" if variant else run_dir.name
                    target = target_algo / name
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.move(str(run_dir), str(target))
                    meta_path = target / "run.json"
                    if meta_path.exists():
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        meta["variant"] = variant
                        meta["path"] = str(target.relative_to(ROOT.parent)).replace("\\", "/")
                        meta_path.write_text(
                            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
                        )
                algo_dir.rmdir()
            variant_dir.rmdir()

    # 2. rebuild the index tables from run.json files
    rows = []
    for meta_path in sorted(ROOT.rglob("run.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        reward = meta.get("test_reward", {})
        rows.append(
            {
                "campaign": meta.get("campaign", ""),
                "algorithm": meta.get("algorithm", ""),
                "variant": meta.get("variant", ""),
                "run_id": meta.get("run_id", ""),
                "env_steps": meta.get("env_steps", 0),
                "test_reward_first": reward.get("first"),
                "test_reward_last": reward.get("last"),
                "test_reward_best": reward.get("best"),
                "test_reward_best_step": reward.get("best_step"),
                "n_eval_points": reward.get("n", 0),
                "ddim_eta_last": meta.get("ddim_eta_last"),
                "source_mb": meta.get("source_mb"),
                "path": meta.get("path", ""),
            }
        )
    rows.sort(key=lambda r: (r["campaign"], r["algorithm"], r["variant"], r["run_id"]))
    with (ROOT / "runs_index.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import statistics

    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["campaign"], row["algorithm"]), []).append(row)
    agg = []
    for (campaign, algorithm), group in sorted(groups.items()):
        lasts = [g["test_reward_last"] for g in group if g["test_reward_last"] is not None]
        bests = [g["test_reward_best"] for g in group if g["test_reward_best"] is not None]
        steps = [g["env_steps"] for g in group if g["env_steps"]]
        agg.append(
            {
                "campaign": campaign,
                "algorithm": algorithm,
                "n_runs": len(group),
                "variants": "|".join(sorted({g["variant"] for g in group if g["variant"]})),
                "env_steps_max": max(steps) if steps else 0,
                "final_mean": round(statistics.fmean(lasts), 3) if lasts else "",
                "final_std": round(statistics.stdev(lasts), 3) if len(lasts) > 1 else 0.0,
                "best_mean": round(statistics.fmean(bests), 3) if bests else "",
                "per_seed_final": "|".join(f"{v:.2f}" for v in lasts),
                "dir": f"data/{campaign}/{algorithm}",
            }
        )
    with (ROOT / "by_algorithm.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(agg[0]))
        writer.writeheader()
        writer.writerows(agg)

    print(f"runs={len(rows)} groups={len(agg)}")
    for row in agg:
        if row["campaign"] in {"fig6-main", "ablation", "learning-rate", "fallback-rerun", "trials"}:
            print(f"  {row['campaign']:16s} {row['algorithm']:10s} n={row['n_runs']} "
                  f"final={row['final_mean']}±{row['final_std']} best={row['best_mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
