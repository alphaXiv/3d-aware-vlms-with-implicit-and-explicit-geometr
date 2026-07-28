# Reproducing Geometry Fusion in 3D-Aware VLMs

## Reproduction result

We tested the central claim of [*3D-Aware VLMs with Implicit and Explicit Geometries* (arXiv:2607.21595)](https://arxiv.org/abs/2607.21595): implicit and explicit geometry should each improve 3D grounding over RGB, and cross-attention should combine them better than simple fusion.

**Assessment: not reproduced in this bounded setup.** The paper’s 3D-video ablation rises from **30.9 F1@0.25** for its baseline to **34.7 explicit**, **40.5 implicit**, and **42.8 combined**. Our public ScanNet/ScanRefer-style proposal-refined test instead measured **17.44% Acc@0.25 RGB**, **13.95% explicit**, **13.63% implicit**, and **12.77% combined IEA** over eight paired seeds. IEA led concatenation by only 0.05 percentage points and addition by 0.54 points, with both paired intervals crossing zero.

This is a deliberate downscale: frozen CLIP and VGGT replace the paper’s jointly trained 3B VLM and AnySplat path; ten public validation scenes are split six/four by scene; and a matched 595,073-parameter adapter selects among real object proposals rather than producing free-form detections. Reconstructed VGGT point maps are the primary explicit stream, with sensor-mesh and shuffled-geometry controls.

Runs used **Kubernetes**, **NVIDIA RTX PRO 6000 Blackwell** GPUs, **16 GPUs peak concurrent**, and **1.6434 elapsed wall hours** from first valid run start to final run finish.

[Read the illustrated report](reports/geometry-reproduction/report.md) · [Inspect the self-contained notebook](notebooks/geometry_reproduction.py) · [Download the measured arrays](reports/geometry-reproduction/results.json)

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/3d-aware-vlms-with-implicit-and-explicit-geometr/blob/main/notebooks/geometry_reproduction.py)

### Experiment log

Every formal node inherited the exact same command. Links for seeds 4–7 point to the robustness-extension branch; seeds 0–3 are on the primary branch.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Public report, notebook, and reproducible code | Not run as an experiment (publication surface) | Presentation only | No experiment |
| [RGB seeds 0–3](https://github.com/alphaXiv/3d-aware-vlms-with-implicit-and-explicit-geometr/tree/orx/verified-ten-scene-rgb) / [4–7](https://github.com/alphaXiv/3d-aware-vlms-with-implicit-and-explicit-geometr/tree/orx/verified-rgb-additional-seeds) | Matched RGB-only control | `bash scripts/run_reproduction.sh` | 17.44% Acc@0.25, eight seeds | Kubernetes; 4× RTX PRO 6000 Blackwell/job |
| [Implicit seeds 0–3](https://github.com/alphaXiv/3d-aware-vlms-with-implicit-and-explicit-geometr/tree/orx/verified-implicit-augmentation) / [4–7](https://github.com/alphaXiv/3d-aware-vlms-with-implicit-and-explicit-geometr/tree/orx/verified-implicit-additional-seeds) | Add frozen VGGT implicit tokens | `bash scripts/run_reproduction.sh` | 13.63%; −3.81 points vs RGB | Kubernetes; 4× RTX PRO 6000 Blackwell/job |
| [Explicit seeds 0–3](https://github.com/alphaXiv/3d-aware-vlms-with-implicit-and-explicit-geometr/tree/orx/verified-explicit-reconstruction) / [4–7](https://github.com/alphaXiv/3d-aware-vlms-with-implicit-and-explicit-geometr/tree/orx/verified-explicit-additional-seeds) | Add VGGT-reconstructed point maps | `bash scripts/run_reproduction.sh` | 13.95%; −3.49 points vs RGB | Kubernetes; 4× RTX PRO 6000 Blackwell/job |
| [IEA seeds 0–3](https://github.com/alphaXiv/3d-aware-vlms-with-implicit-and-explicit-geometr/tree/orx/verified-combined-iea) / [4–7](https://github.com/alphaXiv/3d-aware-vlms-with-implicit-and-explicit-geometr/tree/orx/verified-iea-additional-seeds) | Cross-attend implicit to explicit tokens | `bash scripts/run_reproduction.sh` | 12.77%; −4.67 points vs RGB | Kubernetes; 4× RTX PRO 6000 Blackwell/job |
| [Concat seeds 0–3](https://github.com/alphaXiv/3d-aware-vlms-with-implicit-and-explicit-geometr/tree/orx/verified-fusion-concatenation) / [4–7](https://github.com/alphaXiv/3d-aware-vlms-with-implicit-and-explicit-geometr/tree/orx/verified-concat-additional-seeds) | Simple concatenation fusion | `bash scripts/run_reproduction.sh` | 12.71%; effectively tied with IEA | Kubernetes; 4× RTX PRO 6000 Blackwell/job |
| [Addition seeds 0–3](https://github.com/alphaXiv/3d-aware-vlms-with-implicit-and-explicit-geometr/tree/orx/verified-fusion-addition) / [4–7](https://github.com/alphaXiv/3d-aware-vlms-with-implicit-and-explicit-geometr/tree/orx/verified-addition-additional-seeds) | Simple elementwise addition | `bash scripts/run_reproduction.sh` | 12.23%; IEA lead uncertain | Kubernetes; 4× RTX PRO 6000 Blackwell/job |
| [Shuffled seeds 0–3](https://github.com/alphaXiv/3d-aware-vlms-with-implicit-and-explicit-geometr/tree/orx/verified-shuffled-explicit-control) / [4–7](https://github.com/alphaXiv/3d-aware-vlms-with-implicit-and-explicit-geometr/tree/orx/verified-shuffled-explicit-additional-seeds) | Break explicit scene correspondence | `bash scripts/run_reproduction.sh` | 14.22%; +1.45 points over aligned IEA | Kubernetes; 4× RTX PRO 6000 Blackwell/job |
| [Sensor geometry](https://github.com/alphaXiv/3d-aware-vlms-with-implicit-and-explicit-geometr/tree/orx/verified-sensor-mesh-control) | Replace reconstructed points with privileged mesh samples | `bash scripts/run_reproduction.sh` | 14.59%, four seeds; still below RGB | Kubernetes; 4× RTX PRO 6000 Blackwell |

## Run the protocol

The formal environment is pinned in [requirements-repro.txt](requirements-repro.txt), the matched configuration is in [configs/variant.json](configs/variant.json), and Kubernetes resources are declared in [.orx/k8s.yaml](.orx/k8s.yaml). The public data and model weights are fetched at run time; none are redistributed in this repository.

```bash
bash scripts/run_reproduction.sh
```

The script emits machine-readable `ORX_EVENT` records for source hashes, per-seed curves, per-scene predictions, throughput, parameter count, GPU identity, and terminal metrics. To explore the already-completed evidence locally:

```bash
python -m marimo edit notebooks/geometry_reproduction.py
```
