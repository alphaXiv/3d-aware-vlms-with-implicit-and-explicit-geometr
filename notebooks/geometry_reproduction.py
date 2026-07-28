import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import statistics

    import marimo as mo

    return math, mo, statistics


@app.cell
def _():
    # Embedded terminal evidence: opening this notebook never reruns training.
    results = {
        "RGB": {
            "acc25": [0.1673819742, 0.1845493562, 0.1502145923, 0.2060085837, 0.1502145923, 0.1416309013, 0.1974248927, 0.1974248927],
            "acc50": [0.1587982833, 0.1673819742, 0.1373390558, 0.2060085837, 0.1502145923, 0.1416309013, 0.1974248927, 0.1888412017],
        },
        "Implicit": {
            "acc25": [0.1459227468, 0.1201716738, 0.1459227468, 0.1287553648, 0.1072961373, 0.1716738197, 0.1545064378, 0.1158798283],
            "acc50": [0.1459227468, 0.1201716738, 0.1459227468, 0.1287553648, 0.1072961373, 0.1716738197, 0.1545064378, 0.1158798283],
        },
        "Explicit": {
            "acc25": [0.1201716738, 0.1673819742, 0.1502145923, 0.1330472103, 0.1545064378, 0.0987124464, 0.1673819742, 0.1244635193],
            "acc50": [0.1201716738, 0.1673819742, 0.1502145923, 0.1330472103, 0.1545064378, 0.0987124464, 0.1630901288, 0.1201716738],
        },
        "IEA": {
            "acc25": [0.1759656652, 0.1115879828, 0.1330472103, 0.1287553648, 0.1201716738, 0.1416309013, 0.0858369099, 0.1244635193],
            "acc50": [0.1673819742, 0.1115879828, 0.1330472103, 0.1287553648, 0.1201716738, 0.1416309013, 0.0858369099, 0.1201716738],
        },
        "Concat": {
            "acc25": [0.1201716738, 0.1545064378, 0.1416309013, 0.1502145923, 0.1072961373, 0.0643776824, 0.1244635193, 0.1545064378],
            "acc50": [0.1115879828, 0.1545064378, 0.1416309013, 0.1502145923, 0.1072961373, 0.0643776824, 0.1244635193, 0.1545064378],
        },
        "Addition": {
            "acc25": [0.1244635193, 0.1030042918, 0.1545064378, 0.1072961373, 0.1587982833, 0.0944206009, 0.0944206009, 0.1416309013],
            "acc50": [0.1244635193, 0.1030042918, 0.1545064378, 0.1072961373, 0.1587982833, 0.0944206009, 0.0944206009, 0.1416309013],
        },
        "Shuffled": {
            "acc25": [0.1802575107, 0.1545064378, 0.1459227468, 0.1630901288, 0.1330472103, 0.1330472103, 0.1072961373, 0.1201716738],
            "acc50": [0.1716738197, 0.1502145923, 0.1459227468, 0.1630901288, 0.1330472103, 0.1287553648, 0.1072961373, 0.1158798283],
        },
        "Sensor": {
            "acc25": [0.1416309013, 0.1244635193, 0.1416309013, 0.1759656652],
            "acc50": [0.1330472103, 0.1201716738, 0.1416309013, 0.1759656652],
        },
    }
    return (results,)


@app.cell
def _(mo):
    mo.md(r"""
    # Geometry tokens for 3D grounding: an executable walkthrough

    A vision–language model can recognize objects yet still have a weak sense of where they are in 3D. The paper [*3D-Aware VLMs with Implicit and Explicit Geometries*](https://arxiv.org/abs/2607.21595) reports that learned **implicit** scene tokens and **explicit** reconstructed 3D points each help, and that cross-attention (IEA) combines them best.

    This notebook explains a fresh, bounded reproduction on real public ScanNet/ScanRefer-style scenes. It embeds the completed Kubernetes evidence, so interacting below recomputes summaries only—**it does not rerun training**.

    **Verdict:** the target claims were **not reproduced in this frozen-encoder, proposal-refined adapter setup**. RGB-only led every geometry condition.
    """)
    return


@app.cell
def _(mo):
    metric_picker = mo.ui.dropdown(
        options=["Acc@0.25", "Acc@0.50"],
        value="Acc@0.25",
        label="Held-out metric",
    )
    metric_picker
    return (metric_picker,)


@app.cell
def _(metric_picker, mo, results, statistics):
    metric_key = "acc25" if metric_picker.value == "Acc@0.25" else "acc50"
    order = ["RGB", "Implicit", "Explicit", "IEA", "Concat", "Addition", "Shuffled", "Sensor"]
    colors = {
        "RGB": "#31688e",
        "Implicit": "#e6c700",
        "Explicit": "#35b779",
        "IEA": "#7e03a8",
        "Concat": "#6f9fbb",
        "Addition": "#ef6c35",
        "Shuffled": "#d9485f",
        "Sensor": "#2a9d8f",
    }
    metric_means = {name: statistics.mean(results[name][metric_key]) * 100 for name in order}
    bars = []
    for _index, _name in enumerate(order):
        _value = metric_means[_name]
        _y = 42 + _index * 46
        _width = _value / 22 * 560
        bars.append(
            f'<text x="15" y="{_y + 21}" font-size="15">{_name}</text>'
            f'<rect x="120" y="{_y}" width="{_width:.1f}" height="29" rx="4" fill="{colors[_name]}"/>'
            f'<text x="{128 + _width:.1f}" y="{_y + 20}" font-size="14" font-weight="700">{_value:.2f}%</text>'
        )
    chart_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 790 430" role="img">'
        '<rect width="790" height="430" fill="white"/>'
        f'<text x="15" y="25" font-size="19" font-weight="700">{metric_picker.value} across conditions</text>'
        + "".join(bars)
        + "</svg>"
    )
    mo.Html(chart_svg)
    return metric_key, metric_means


@app.cell
def _(metric_key, metric_means, mo, results, statistics):
    rows = []
    for _condition in ["RGB", "Implicit", "Explicit", "IEA", "Concat", "Addition", "Shuffled", "Sensor"]:
        _values = results[_condition][metric_key]
        rows.append(
            f"| {_condition} | {len(_values)} | {metric_means[_condition]:.2f}% | "
            f"{statistics.stdev(_values) * 100:.2f} |"
        )
    mo.md(
        "## The measured ordering\n\n"
        "| Condition | Seeds | Mean | Seed SD (points) |\n"
        "|---|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n\nAll conditions use **595,073 trainable parameters**. "
        "The sensor control has four seeds; all other rows have eight."
    )
    return


@app.cell
def _(math, mo, results, statistics):
    def paired_effect(first: str, second: str):
        differences = [
            (a - b) * 100
            for a, b in zip(results[first]["acc25"], results[second]["acc25"])
        ]
        average = statistics.mean(differences)
        half_width = 2.364624251 * statistics.stdev(differences) / math.sqrt(len(differences))
        return average, average - half_width, average + half_width

    comparisons = [
        ("Implicit − RGB", *paired_effect("Implicit", "RGB")),
        ("Explicit − RGB", *paired_effect("Explicit", "RGB")),
        ("IEA − RGB", *paired_effect("IEA", "RGB")),
        ("IEA − concat", *paired_effect("IEA", "Concat")),
        ("IEA − addition", *paired_effect("IEA", "Addition")),
        ("IEA − shuffled", *paired_effect("IEA", "Shuffled")),
    ]
    effect_rows = "\n".join(
        f"| {label} | {average:+.2f} | [{low:+.2f}, {high:+.2f}] |"
        for label, average, low, high in comparisons
    )
    mo.md(
        f"""
        ## Paired effects answer the claims

        Each seed index is paired across conditions. Intervals are two-sided 95% Student-*t* intervals over eight paired differences.

        | Contrast | Mean difference (points) | 95% interval |
        |---|---:|---:|
        {effect_rows}

        The implicit and explicit differences from RGB are negative and exclude zero. IEA is statistically unresolved against simple fusion, while shuffled scene geometry is numerically better than aligned IEA. The data therefore do not support a useful, correspondence-specific geometry benefit in this setup.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## From pixels to a grounded box

        1. **Freeze the encoders.** CLIP turns eight RGB frames into spatial image tokens and the referring expression into a text vector. VGGT turns the same frames into learned scene tokens and a reconstructed point map.
        2. **Match capacity.** Every variant instantiates the same 595,073-parameter adapter; unused modules remain registered, so parameter count cannot explain a difference.
        3. **Change only the feature path.** RGB is the control. Implicit and explicit variants add one geometry stream. IEA lets implicit tokens query explicit tokens. Concatenation and addition are simple-fusion controls.
        4. **Ground on unseen scenes.** A text-conditioned scorer chooses among real object boxes in four held-out scenes. Acc@0.25 and Acc@0.50 test 3D box overlap.

        The public split contains six train scenes, four held-out scenes, 328 post-validation training expressions, and 233 held-out expressions. Primary explicit evidence comes from RGB-reconstructed VGGT points; a sensor-mesh branch checks the privileged-depth confound.

        ## Compute and provenance

    The queue runner verified **18 successful Kubernetes runs** on **NVIDIA RTX PRO 6000 Blackwell** GPUs, with a peak of **16 concurrent GPUs** and **1.955439 observed campaign wall hours**. Each formal job used four GPUs and four seeds. The exact shared command was:

        ```bash
        bash scripts/run_reproduction.sh
        ```

        Explore the [detailed report](https://github.com/alphaXiv/3d-aware-vlms-with-implicit-and-explicit-geometr/blob/main/reports/geometry-reproduction/report.md), [raw embedded-result source](https://github.com/alphaXiv/3d-aware-vlms-with-implicit-and-explicit-geometr/blob/main/reports/geometry-reproduction/results.json), or the [matched adapter implementation](https://github.com/alphaXiv/3d-aware-vlms-with-implicit-and-explicit-geometr/blob/main/src/reproduce.py).

        ## What this does—and does not—establish

        This is real held-out public-scene evidence, not a synthetic proxy. But it is a small proposal-refined grounding task with frozen CLIP/VGGT, a local split of public validation scenes, and no 3B language model. It does **not** test the paper’s full multi-task training recipe or directly compare metric magnitudes with its 3D-video F1 table. A faithful full-scale reproduction still needs the authors’ AnySplat checkpoints and data preparation, official ScanRefer assets, and joint 3B VLM training.
    """)
    return


if __name__ == "__main__":
    app.run()
