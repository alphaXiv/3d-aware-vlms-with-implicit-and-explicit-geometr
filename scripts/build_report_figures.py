"""Build the report's dependency-free SVG figures from the measured results."""

from __future__ import annotations

import math
from pathlib import Path


OUT = Path(__file__).parents[1] / "reports" / "geometry-reproduction" / "images"
INK = "#17202a"
MUTED = "#667085"
GRID = "#d8dee8"
RGB = "#31688e"
EXPLICIT = "#35b779"
IMPLICIT = "#fde725"
IEA = "#7e03a8"
CONTROL = "#ef6c35"

ACC25 = {
    "RGB": [16.7382, 18.4549, 15.0215, 20.6009, 15.0215, 14.1631, 19.7425, 19.7425],
    "Implicit": [14.5923, 12.0172, 14.5923, 12.8755, 10.7296, 17.1674, 15.4506, 11.5880],
    "Explicit": [12.0172, 16.7382, 15.0215, 13.3047, 15.4506, 9.8712, 16.7382, 12.4464],
    "IEA": [17.5966, 11.1588, 13.3047, 12.8755, 12.0172, 14.1631, 8.5837, 12.4464],
    "Concat": [12.0172, 15.4506, 14.1631, 15.0215, 10.7296, 6.4378, 12.4464, 15.4506],
    "Addition": [12.4464, 10.3004, 15.4506, 10.7296, 15.8798, 9.4421, 9.4421, 14.1631],
    "Shuffled": [18.0258, 15.4506, 14.5923, 16.3090, 13.3047, 13.3047, 10.7296, 12.0172],
    "Sensor": [14.1631, 12.4464, 14.1631, 17.5966],
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def header(title: str, subtitle: str) -> list[str]:
    return [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="560" viewBox="0 0 1000 560" role="img">',
        f"<title>{title}</title>",
        "<style>text{font-family:Inter,Arial,sans-serif;fill:#17202a}.title{font-size:25px;font-weight:700}.sub{font-size:15px;fill:#667085}.axis{font-size:13px;fill:#667085}.label{font-size:15px;font-weight:600}.value{font-size:14px;font-weight:700}.note{font-size:13px;fill:#667085}.grid{stroke:#d8dee8;stroke-width:1}.zero{stroke:#17202a;stroke-width:2}.point{stroke:#fff;stroke-width:1.5}</style>",
        '<rect width="1000" height="560" fill="#ffffff" rx="12"/>',
        f'<text x="40" y="42" class="title">{title}</text>',
        f'<text x="40" y="68" class="sub">{subtitle}</text>',
    ]


def finish(lines: list[str], name: str) -> None:
    lines.append("</svg>")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("\n".join(lines) + "\n")


def headline() -> None:
    lines = header(
        "The paper's monotonic geometry gain did not appear here",
        "Separate axes: paper 3D-video F1@0.25 (left); reproduction ScanRefer-style Acc@0.25 (right).",
    )
    panels = [
        (50, "Paper ablation", [("RGB baseline", 30.9, RGB), ("+ explicit", 34.7, EXPLICIT), ("+ implicit", 40.5, IMPLICIT), ("IEA combined", 42.8, IEA)], 50),
        (525, "Bounded reproduction · 8 seeds", [("RGB", 17.44, RGB), ("+ explicit", 13.95, EXPLICIT), ("+ implicit", 13.63, IMPLICIT), ("IEA combined", 12.77, IEA)], 22),
    ]
    for x0, title, values, xmax in panels:
        lines.append(f'<text x="{x0}" y="108" class="label">{title}</text>')
        for tick in range(0, xmax + 1, 10 if xmax > 30 else 5):
            x = x0 + 120 + tick / xmax * 300
            lines.append(f'<line x1="{x:.1f}" y1="128" x2="{x:.1f}" y2="462" class="grid"/>')
            lines.append(f'<text x="{x:.1f}" y="486" text-anchor="middle" class="axis">{tick}</text>')
        for i, (label, value, color) in enumerate(values):
            y = 148 + i * 78
            width = value / xmax * 300
            lines.append(f'<text x="{x0}" y="{y + 24}" class="label">{label}</text>')
            lines.append(f'<rect x="{x0 + 120}" y="{y}" width="{width:.1f}" height="34" rx="5" fill="{color}"/>')
            lines.append(f'<text x="{x0 + 128 + width:.1f}" y="{y + 23}" class="value">{value:.2f}</text>')
    lines.append('<text x="500" y="530" text-anchor="middle" class="note">Higher is better. Values are directional comparisons, not a shared metric scale.</text>')
    finish(lines, "headline.svg")


def paired_effects() -> None:
    lines = header(
        "Paired effects do not support a useful geometry advantage",
        "Mean Acc@0.25 difference in percentage points; whiskers are 95% t intervals over 8 paired seeds.",
    )
    effects = [
        ("Implicit − RGB", -3.81, -7.00, -0.61, IMPLICIT),
        ("Explicit − RGB", -3.49, -5.98, -1.00, EXPLICIT),
        ("IEA − RGB", -4.67, -8.26, -1.08, IEA),
        ("IEA − concat", 0.05, -3.69, 3.79, IEA),
        ("IEA − addition", 0.54, -2.20, 3.27, IEA),
        ("IEA − shuffled", -1.45, -2.95, 0.05, CONTROL),
    ]
    x0, x1, lo, hi = 300, 930, -9, 5
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    for tick in range(-8, 6, 2):
        x = sx(tick)
        lines.append(f'<line x1="{x:.1f}" y1="108" x2="{x:.1f}" y2="486" class="{"zero" if tick == 0 else "grid"}"/>')
        lines.append(f'<text x="{x:.1f}" y="512" text-anchor="middle" class="axis">{tick:+d}</text>')
    for i, (label, value, low, high, color) in enumerate(effects):
        y = 142 + i * 58
        lines.append(f'<text x="42" y="{y + 5}" class="label">{label}</text>')
        lines.append(f'<line x1="{sx(low):.1f}" y1="{y}" x2="{sx(high):.1f}" y2="{y}" stroke="{color}" stroke-width="5" stroke-linecap="round"/>')
        lines.append(f'<circle cx="{sx(value):.1f}" cy="{y}" r="8" fill="{color}" class="point"/>')
        lines.append(f'<text x="947" y="{y + 5}" class="value" text-anchor="end">{value:+.2f}</text>')
    lines.append('<text x="615" y="544" text-anchor="middle" class="note">Positive favors the first-named condition; intervals crossing zero are inconclusive.</text>')
    finish(lines, "paired-effects.svg")


def fusion_seeds() -> None:
    lines = header(
        "Fusion methods overlap across seeds",
        "Each dot is one training seed; diamonds mark the 8-seed mean on the same held-out scenes.",
    )
    names = [("IEA", IEA), ("Concat", RGB), ("Addition", CONTROL)]
    y0, y1, ymin, ymax = 440, 115, 5, 20
    sy = lambda v: y0 - (v - ymin) / (ymax - ymin) * (y0 - y1)
    for tick in range(5, 21, 5):
        y = sy(tick)
        lines.append(f'<line x1="120" y1="{y:.1f}" x2="920" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="98" y="{y + 5:.1f}" text-anchor="end" class="axis">{tick}%</text>')
    for j, (name, color) in enumerate(names):
        x = 280 + j * 230
        vals = ACC25[name]
        for i, value in enumerate(vals):
            jitter = ((i % 4) - 1.5) * 16
            lines.append(f'<circle cx="{x + jitter:.1f}" cy="{sy(value):.1f}" r="7" fill="{color}" fill-opacity=".68" class="point"/>')
        m = mean(vals)
        points = f"{x},{sy(m)-11:.1f} {x+11},{sy(m):.1f} {x},{sy(m)+11:.1f} {x-11},{sy(m):.1f}"
        lines.append(f'<polygon points="{points}" fill="{INK}"/>')
        lines.append(f'<text x="{x}" y="478" text-anchor="middle" class="label">{name}</text>')
        lines.append(f'<text x="{x}" y="505" text-anchor="middle" class="value">mean {m:.2f}%</text>')
    lines.append('<text x="500" y="540" text-anchor="middle" class="note">IEA leads concat by 0.05 points and addition by 0.54 points—well within paired uncertainty.</text>')
    finish(lines, "fusion-seeds.svg")


def diagnostic() -> None:
    lines = header(
        "Correct correspondence was not the source of improvement",
        "Acc@0.25 means; aligned, shuffled, and sensor-geometry controls use the same matched adapter.",
    )
    values = [("RGB", mean(ACC25["RGB"]), RGB, 8), ("Aligned IEA", mean(ACC25["IEA"]), IEA, 8), ("Shuffled explicit", mean(ACC25["Shuffled"]), CONTROL, 8), ("Sensor geometry", mean(ACC25["Sensor"]), EXPLICIT, 4)]
    y0, y1, ymax = 440, 118, 22
    sy = lambda v: y0 - v / ymax * (y0 - y1)
    for tick in range(0, 23, 5):
        y = sy(tick)
        lines.append(f'<line x1="95" y1="{y:.1f}" x2="940" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="78" y="{y + 5:.1f}" text-anchor="end" class="axis">{tick}%</text>')
    for i, (label, value, color, n) in enumerate(values):
        x = 145 + i * 205
        height = y0 - sy(value)
        lines.append(f'<rect x="{x}" y="{sy(value):.1f}" width="125" height="{height:.1f}" rx="6" fill="{color}"/>')
        lines.append(f'<text x="{x+62.5}" y="{sy(value)-10:.1f}" text-anchor="middle" class="value">{value:.2f}%</text>')
        lines.append(f'<text x="{x+62.5}" y="474" text-anchor="middle" class="label">{label}</text>')
        lines.append(f'<text x="{x+62.5}" y="498" text-anchor="middle" class="axis">n={n} seeds</text>')
    lines.append('<text x="500" y="540" text-anchor="middle" class="note">Shuffling explicit scene tokens is +1.45 points over aligned IEA; sensor geometry remains below RGB.</text>')
    finish(lines, "correspondence-diagnostic.svg")


if __name__ == "__main__":
    headline()
    paired_effects()
    fusion_seeds()
    diagnostic()
