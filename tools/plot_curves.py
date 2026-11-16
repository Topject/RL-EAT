"""Redraw every figure in this repository from `data/` with Pillow only.

No matplotlib is required: the plots are drawn on a 2x canvas and downsampled,
which gives clean anti-aliased lines and CJK labels (Microsoft YaHei).

    python tools/plot_curves.py --data data --out figures/curves

Figures produced
    fig6_convergence.png       the seven Figure-6 algorithms, mean of 3 seeds
    fig6_final_bar.png         final validation reward per algorithm
    <algo>_convergence.png     each algorithm, its 3 seeds + mean
    fig6_eta.png               EAT's learned DDIM eta over training
    ablation.png               SAC-CA / D2SAC-SA / EAT-CA against their parents
    learning_rate.png          3e-4 arms against the 1e-4 main sweep
    fallback_rerun.png         the fallback env sweep, all seven algorithms
    scale_density.png          SAC / EAT across node count and task density
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCALE = 2
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhl.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]
INK = (30, 36, 44)
MUTED = (108, 118, 130)
GRID = (228, 232, 238)
FRAME = (198, 205, 213)
PAPER = (255, 255, 255)

PALETTE = [
    (31, 119, 180), (214, 84, 60), (44, 160, 44), (150, 90, 180),
    (230, 150, 20), (60, 175, 190), (170, 60, 120), (110, 110, 110),
]

ALGO_STYLE = {
    "ddpg": ((122, 122, 122), "DDPG"),
    "td3": ((150, 90, 180), "TD3"),
    "sac": ((44, 160, 44), "SAC"),
    "ppo": ((31, 119, 180), "PPO"),
    "d2sac": ((230, 150, 20), "D2SAC"),
    "dppo": ((214, 84, 60), "DPPO"),
    "eat": ((196, 30, 58), "EAT"),
    "eat_ca": ((226, 116, 134), "EAT-CA"),
    "eat3e": ((226, 116, 134), "EAT (lr 3e-4)"),
    "d2sac3e": ((244, 196, 128), "D2SAC (lr 3e-4)"),
    "d2sac_sa": ((244, 196, 128), "D2SAC-SA"),
    "sac_ca": ((140, 200, 140), "SAC-CA"),
    "td33e": ((190, 160, 210), "TD3 (lr 3e-4)"),
    "tryeat": ((196, 30, 58), "tryeat"),
}

ORDER = ["ddpg", "td3", "sac", "ppo", "d2sac", "dppo", "eat"]
_FONTS: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}

Y_LABEL = "\u9a8c\u8bc1\u56de\u62a5 / validation return"
X_LABEL = "\u73af\u5883\u6b65\u6570 / environment steps"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key in _FONTS:
        return _FONTS[key]
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                face = ImageFont.truetype(path, size * SCALE, index=1 if bold else 0)
                _FONTS[key] = face
                return face
            except Exception:  # noqa: BLE001
                continue
    fallback = ImageFont.load_default()
    _FONTS[key] = fallback
    return fallback


class Figure:
    """Absolute-layout canvas: margins in CSS pixels, plot box in the middle."""

    def __init__(self, width: int = 1080, height: int = 660,
                 left: int = 118, right: int = 34, top: int = 104, bottom: int = 96) -> None:
        self.w, self.h = width, height
        self.img = Image.new("RGB", (width * SCALE, height * SCALE), PAPER)
        self.layer = Image.new("RGBA", (width * SCALE, height * SCALE), (0, 0, 0, 0))
        self.ld = ImageDraw.Draw(self.layer)
        self.box = (left, top, width - right, height - bottom)

    def text(self, x: float, y: float, s: str, color=INK, size=13, anchor="la", bold=False):
        ImageDraw.Draw(self.img).text(
            (x * SCALE, y * SCALE), s, font=font(size, bold), fill=color, anchor=anchor
        )

    def vtext(self, x: float, y: float, s: str, color=INK, size=13):
        face = font(size)
        probe = Image.new("RGBA", (size * SCALE * 4, size * SCALE * 4), (0, 0, 0, 0))
        ImageDraw.Draw(probe).text((size * SCALE, size * SCALE), s, font=face, fill=color)
        bbox = probe.getbbox()
        if bbox is None:
            return
        tmp = probe.crop(bbox).rotate(90, expand=True)
        self.img.paste(tmp, (int(x * SCALE - tmp.width / 2), int(y * SCALE - tmp.height / 2)), tmp)

    def line_abs(self, pts, color, width=2):
        self.ld.line([(x * SCALE, y * SCALE) for x, y in pts], fill=color,
                     width=int(width * SCALE), joint="curve")

    def swatch(self, x0: float, x1: float, y: float, color, width=3):
        """Legend key, painted straight onto the page so save()'s clip cannot eat it."""
        ImageDraw.Draw(self.img).line(
            [(x0 * SCALE, y * SCALE), (x1 * SCALE, y * SCALE)],
            fill=color, width=int(width * SCALE),
        )

    def set_ranges(self, xmax: float, ymin: float, ymax: float):
        self.xmax, self.ymin, self.ymax = xmax, ymin, ymax
        self.x0, self.y0, self.x1, self.y1 = self.box

    def px(self, x: float) -> float:
        return self.x0 + (x / self.xmax) * (self.x1 - self.x0)

    def py(self, y: float) -> float:
        frac = (y - self.ymin) / (self.ymax - self.ymin) if self.ymax > self.ymin else 0.0
        return self.y1 - frac * (self.y1 - self.y0)

    def plot_line(self, pts, color, width=2):
        if len(pts) < 2:
            return
        self.ld.line([(self.px(x) * SCALE, self.py(y) * SCALE) for x, y in pts], fill=color,
                     width=max(1, int(width * SCALE)), joint="curve")

    def band(self, xs, lo, hi, color, alpha=40):
        poly = [(self.px(x) * SCALE, self.py(v) * SCALE) for x, v in zip(xs, hi)]
        poly += [(self.px(x) * SCALE, self.py(v) * SCALE) for x, v in zip(reversed(xs), reversed(lo))]
        self.ld.polygon(poly, fill=(color[0], color[1], color[2], alpha))

    def bar(self, cx: float, half: float, top: float, color):
        self.ld.rectangle(
            [self.px(cx - half) * SCALE, self.py(top) * SCALE,
             self.px(cx + half) * SCALE, self.py(self.ymin) * SCALE],
            fill=color,
        )

    def frame(self, x_ticks, y_ticks, x_fmt=str, y_fmt=str):
        d = ImageDraw.Draw(self.img)
        d.rectangle([self.x0 * SCALE, self.y0 * SCALE, self.x1 * SCALE, self.y1 * SCALE],
                    outline=FRAME, width=SCALE)
        for y in y_ticks:
            yy = self.py(y)
            if not (self.y0 - 0.5 <= yy <= self.y1 + 0.5):
                continue
            d.line([self.x0 * SCALE, yy * SCALE, self.x1 * SCALE, yy * SCALE], fill=GRID, width=SCALE)
            self.text(self.x0 - 10, yy, y_fmt(y), MUTED, 12, anchor="rm")
        for x in x_ticks:
            xx = self.px(x)
            if not (self.x0 - 0.5 <= xx <= self.x1 + 0.5):
                continue
            d.line([xx * SCALE, self.y0 * SCALE, xx * SCALE, self.y1 * SCALE], fill=GRID, width=SCALE)
            self.text(xx, self.y1 + 10, x_fmt(x), MUTED, 12, anchor="ma")

    def flush(self):
        self.img = Image.alpha_composite(self.img.convert("RGBA"), self.layer).convert("RGB")
        self.layer = Image.new("RGBA", self.img.size, (0, 0, 0, 0))
        self.ld = ImageDraw.Draw(self.layer)

    def save(self, path: Path):
        # clip data drawing to the plot box so stray points never overwrite the axes
        box = (int(self.box[0] * SCALE), int(self.box[1] * SCALE),
               int(self.box[2] * SCALE), int(self.box[3] * SCALE))
        mask = Image.new("L", self.layer.size, 0)
        ImageDraw.Draw(mask).rectangle(box, fill=255)
        clipped = Image.new("RGBA", self.layer.size, (0, 0, 0, 0))
        clipped.paste(self.layer, (0, 0), mask)
        self.layer = clipped
        self.flush()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.img.resize((self.w, self.h), Image.LANCZOS).save(path, "PNG", optimize=True)


def header(fig: Figure, title: str, subtitle: str,
           legend_rows: list[list[tuple[str, tuple[int, int, int]]]],
           x_label: str = X_LABEL) -> None:
    fig.text(118, 24, title, INK, 20, bold=True)
    fig.text(120, 56, subtitle, MUTED, 12)
    fig.vtext(50, (fig.box[1] + fig.box[3]) / 2, Y_LABEL, INK, 13)
    fig.text((fig.box[0] + fig.box[2]) / 2, fig.box[3] + 44, x_label, INK, 13, anchor="ma")
    x, y = 120.0, 80.0
    for row in legend_rows:
        for label, color in row:
            fig.swatch(x, x + 26, y, color, 3)
            fig.text(x + 32, y, label, INK, 12, anchor="lm")
            x += 44 + 7.0 * len(label)
            if x > fig.w - 200:
                x, y = 120.0, y + 19.0


def load_curves(run_dir: Path) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = {}
    path = run_dir / "curves.csv"
    if not path.exists():
        return series
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) != 3:
                continue
            tag, step, value = row
            try:
                series.setdefault(tag, []).append((float(step), float(value)))
            except ValueError:
                continue
    for tag in series:
        series[tag].sort(key=lambda p: p[0])
    return series


def ema(points: list[tuple[float, float]], alpha: float = 0.25) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    prev = None
    for step, value in points:
        prev = value if prev is None else alpha * value + (1 - alpha) * prev
        out.append((step, prev))
    return out


def drop_cold_start(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The step-0 evaluation runs before any update, so its 0 return is not a data point."""
    while len(points) > 1 and points[0][1] <= 0.0:
        points = points[1:]
    return points


def align(series_list: list[list[tuple[float, float]]], n: int = 240):
    top = min(s[-1][0] for s in series_list)
    grid = [top * i / (n - 1) for i in range(n)]
    rows = []
    for series in series_list:
        xs = [p[0] for p in series]
        ys = [p[1] for p in series]
        j = 0
        row = []
        for x in grid:
            while j + 1 < len(xs) and xs[j + 1] < x:
                j += 1
            if x <= xs[0]:
                row.append(ys[0])
            elif x >= xs[-1]:
                row.append(ys[-1])
            else:
                x0, x1 = xs[j], xs[j + 1]
                y0, y1 = ys[j], ys[j + 1]
                t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
                row.append(y0 + t * (y1 - y0))
        rows.append(row)
    return grid, rows


def load_runs(data: Path, campaign: str) -> dict[str, list[Path]]:
    base = data / campaign
    out: dict[str, list[Path]] = {}
    if not base.exists():
        return out
    for algo_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        runs = [p.parent for p in sorted(algo_dir.glob("*/curves.csv"))]
        if runs:
            out[algo_dir.name] = runs
    return out


def reward_series(run_dir: Path, tag: str = "test/reward") -> list[tuple[float, float]]:
    return ema(drop_cold_start(load_curves(run_dir).get(tag, [])))


def nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    span = hi - lo
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    step = mag * 10
    for mult in (1, 2, 2.5, 5, 10):
        if span / (mag * mult) <= target + 1:
            step = mag * mult
            break
    start = math.ceil(lo / step) * step
    ticks = []
    v = start
    while v <= hi + 1e-9:
        ticks.append(round(v, 6))
        v += step
    return ticks


def multi_curve(path: Path, title: str, subtitle: str,
                groups: list[tuple[str, list[Path], tuple[int, int, int]]],
                ymin: float, ymax: float, xmax: float, legend_rows: int = 2,
                band_alpha: int = 40) -> None:
    fig = Figure()
    fig.set_ranges(xmax, ymin, ymax)
    items = [(label, color) for label, _, color in groups]
    rows = [items[i::legend_rows] for i in range(min(legend_rows, len(items)))]
    header(fig, title, subtitle, rows)
    fig.frame([xmax * i / 6 for i in range(7)], nice_ticks(ymin, ymax),
              x_fmt=lambda v: f"{v/1000:.0f}k")

    for label, runs, color in groups:
        series = [s for s in (reward_series(r) for r in runs) if len(s) > 1]
        if not series:
            continue
        grid, rows_ = align(series, 240)
        mean = [statistics.fmean(col) for col in zip(*rows_)]
        if len(rows_) > 1:
            fig.band(grid, [min(c) for c in zip(*rows_)], [max(c) for c in zip(*rows_)],
                     color, band_alpha)
        fig.plot_line(list(zip(grid, mean)), color, 2.4)
    fig.save(path)


def bar_chart(path: Path, title: str, subtitle: str,
              rows: list[tuple[str, float, float]], ymax: float, ymin: float = 0.0) -> None:
    fig = Figure(height=620, left=110, bottom=124)
    fig.set_ranges(len(rows), ymin, ymax)
    header(fig, title, subtitle + f" \u00b7 y axis starts at {ymin:g}", [],
           x_label="algorithm / \u7b97\u6cd5")
    fig.frame([], nice_ticks(ymin, ymax))
    slot = fig.xmax / max(len(rows), 1)
    for i, (label, mean, std) in enumerate(rows):
        color = PALETTE[i % len(PALETTE)]
        cx = slot * (i + 0.5)
        fig.bar(cx, slot * 0.3, mean, color)
        fig.plot_line([(cx, mean), (cx, min(ymax, mean + std))], INK, 1.4)
        fig.text(fig.px(cx), fig.py(mean) - 10, f"{mean:.1f}", INK, 13, anchor="mb", bold=True)
        fig.text(fig.px(cx), fig.box[3] + 16, label, INK, 12, anchor="ma")
    fig.text(fig.w - 34, fig.h - 18,
             "bar = mean over seeds, whisker = +1 std", MUTED, 11, anchor="ra")
    fig.save(path)


def eta_figure(path: Path, runs: list[Path], title: str, ymax: float, xmax: float) -> None:
    fig = Figure(height=560)
    fig.set_ranges(xmax, 0.0, ymax)
    fig.text(118, 24, title, INK, 20, bold=True)
    fig.text(120, 56, "eta = sigmoid(eta_param): the learnable randomness of the DDIM sampler",
             MUTED, 12)
    fig.text(120, 80, "seed / \u79cd\u5b50", MUTED, 12)
    fig.text((fig.box[0] + fig.box[2]) / 2, fig.box[3] + 44, X_LABEL, INK, 13, anchor="ma")
    fig.frame([xmax * i / 5 for i in range(6)], nice_ticks(0, ymax),
              x_fmt=lambda v: f"{v/1000:.0f}k")
    y = 100.0
    for i, run in enumerate(runs):
        series = ema(load_curves(run).get("train/ddim_eta", []), 0.2)
        if not series:
            continue
        color = PALETTE[i % len(PALETTE)]
        fig.plot_line(series, color, 2)
        fig.swatch(900, 926, y, color, 3)
        fig.text(932, y, run.name[-8:], INK, 11, anchor="lm")
        y += 18.0
    fig.save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="figures/curves")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []

    fig6 = load_runs(data, "fig6-main")
    groups = [(ALGO_STYLE[a][1], fig6[a], ALGO_STYLE[a][0]) for a in ORDER if a in fig6]
    if groups:
        p = out / "fig6_convergence.png"
        multi_curve(p, "Figure 6 reproduction \u2014 validation return while training",
                    "3 seeds per algorithm \u00b7 EMA(0.25) of test/reward \u00b7 "
                    "band = seed min/max \u00b7 every curve ends at that arm's own step budget",
                    groups, 120, 205, 880_000, legend_rows=3)
        made.append(p)

    for algo in ORDER:
        if algo not in fig6:
            continue
        color, label = ALGO_STYLE[algo]
        runs = fig6[algo]
        top = max(reward_series(r)[-1][0] for r in runs if reward_series(r))
        p = out / f"{algo}_convergence.png"
        multi_curve(p, f"{label} \u2014 three seeds",
                    f"EMA(0.25) of test/reward \u00b7 {len(runs)} runs \u00b7 band = seed min/max",
                    [(label, runs, color)], 120, 205, top * 1.02, legend_rows=1, band_alpha=55)
        made.append(p)

    rows = []
    for algo in ORDER:
        if algo not in fig6:
            continue
        # bar chart uses the raw last measured point, not the EMA, so it matches runs_index.csv
        lasts = [s[-1][1] for s in (drop_cold_start(load_curves(r).get("test/reward", []))
                                    for r in fig6[algo]) if s]
        if not lasts:
            continue
        rows.append((ALGO_STYLE[algo][1], statistics.fmean(lasts),
                     statistics.stdev(lasts) if len(lasts) > 1 else 0.0))
    if rows:
        p = out / "fig6_final_bar.png"
        bar_chart(p, "Final validation return (3 seeds)",
                  "raw last test/reward from data/fig6-main/<algo>/<run>/curves.csv",
                  rows, 200, 140)
        made.append(p)

    if "eat" in fig6:
        p = out / "fig6_eta.png"
        eta_figure(p, fig6["eat"],
                   "EAT \u2014 learned DDIM eta while training", 0.55, 430_000)
        made.append(p)

    abl = load_runs(data, "ablation")
    groups = []
    for algo in ["sac", "sac_ca", "d2sac", "d2sac_sa", "eat", "eat_ca"]:
        pool = abl.get(algo) or fig6.get(algo)
        if not pool:
            continue
        color, label = ALGO_STYLE[algo]
        tag = "\u6d88\u878d\u81c2 / ablation" if algo in {"sac_ca", "d2sac_sa", "eat_ca"} else "parent"
        groups.append((f"{label} ({tag})", pool, color))
    if groups:
        p = out / "ablation.png"
        multi_curve(p, "Ablation \u2014 what each mechanism contributes",
                    "ablation arms (B sweep) plotted beside their parents from the main sweep",
                    groups, 120, 205, 880_000, legend_rows=3)
        made.append(p)

    lr = load_runs(data, "learning-rate")
    groups = []
    for algo in ["d2sac", "d2sac3e", "eat", "eat3e", "td3", "td33e"]:
        pool = lr.get(algo) or fig6.get(algo)
        if not pool:
            continue
        color, label = ALGO_STYLE[algo]
        groups.append((label, pool, color))
    if groups:
        p = out / "learning_rate.png"
        multi_curve(p, "Learning rate \u2014 the 3e-4 arms against the main sweep",
                    "identical protocol; every arm trained to the same 336k-step budget",
                    groups, 120, 205, 880_000, legend_rows=3)
        made.append(p)

    fb = load_runs(data, "fallback-rerun")
    groups = [(ALGO_STYLE[a][1], fb[a], ALGO_STYLE[a][0]) for a in ORDER if a in fb]
    if groups:
        p = out / "fallback_rerun.png"
        multi_curve(p, "Fallback environment sweep",
                    "the second full sweep, after the task-index fallback change in eval",
                    groups, 120, 205, 880_000, legend_rows=3)
        made.append(p)

    sd = load_runs(data, "scale-density-z")
    groups = [(ALGO_STYLE[a][1], sd[a], ALGO_STYLE[a][0]) for a in ("sac", "eat") if a in sd]
    if groups:
        p = out / "scale_density.png"
        multi_curve(p, "Node count and task density",
                    "32-node runs, quality_ratio 0.333 / 0.444 / 0.556 / 0.600",
                    groups, 0, 210, 340_000, legend_rows=1)
        made.append(p)

    for p in made:
        print(f"{p}  {p.stat().st_size/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
