from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from PIL import Image
from plyfile import PlyData
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, CLIPModel


PAPER_ID = "2607.21595"
GPU_MODEL = "NVIDIA RTX PRO 6000 Blackwell"
ANNOTATION_REPO = "zd11024/Video-3D-LLM_data"
ANNOTATION_FILE = "processed/scanrefer_vg_val_llava_style.json"
RGB_REPO = "SpatialVision/scannet_val"
MESH_REPO = "zahidpichen/scannet-dataset"
CLIP_ID = "openai/clip-vit-base-patch32"
VGGT_ID = "facebook/VGGT-1B"
TRAIN_SCENES = [
    "scene0011_00",
    "scene0015_00",
    "scene0030_00",
    "scene0046_00",
    "scene0081_00",
    "scene0084_00",
]
TEST_SCENES = [
    "scene0050_00",
    "scene0063_00",
    "scene0064_00",
    "scene0077_00",
]
ALL_SCENES = TRAIN_SCENES + TEST_SCENES


def emit(event: str, **payload: Any) -> None:
    print("ORX_EVENT " + json.dumps({"event": event, **payload}, sort_keys=True), flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(repo: str, filename: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
    )


def prepare_assets(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Any] = {
        "annotation_repo": ANNOTATION_REPO,
        "rgb_repo": RGB_REPO,
        "mesh_repo": MESH_REPO,
        "clip_model": CLIP_ID,
        "vggt_model": VGGT_ID,
        "files": {},
    }
    ann_src = download_file(ANNOTATION_REPO, ANNOTATION_FILE)
    ann_dst = root / "scanrefer_vg_val_llava_style.json"
    if not ann_dst.exists():
        shutil.copy2(ann_src, ann_dst)
    sources["files"][ANNOTATION_FILE] = sha256_file(ann_dst)

    for scene in ALL_SCENES:
        scene_dir = root / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        entries = list(
            api.list_repo_tree(
                repo_id=RGB_REPO,
                path_in_repo=scene,
                repo_type="dataset",
                recursive=False,
            )
        )
        rgb_files = sorted(
            entry.path for entry in entries if getattr(entry, "path", "").endswith(".jpg")
        )
        if len(rgb_files) < 8:
            raise RuntimeError(f"{scene} has only {len(rgb_files)} public RGB frames")
        indices = np.linspace(0, len(rgb_files) - 1, 8).round().astype(int)
        selected_rgb = [rgb_files[i] for i in indices]
        rgb_hashes = []
        for rgb_name in selected_rgb:
            rgb_src = download_file(RGB_REPO, rgb_name)
            rgb_dst = scene_dir / Path(rgb_name).name
            if not rgb_dst.exists():
                shutil.copy2(rgb_src, rgb_dst)
            rgb_hashes.append(sha256_file(rgb_dst))
        sources["files"][f"{scene}/selected_rgb8"] = hashlib.sha256(
            "".join(rgb_hashes).encode()
        ).hexdigest()

        mesh_name = f"{scene}/{scene}_vh_clean_2.ply"
        mesh_src = download_file(MESH_REPO, mesh_name)
        mesh_dst = scene_dir / f"{scene}_vh_clean_2.ply"
        if not mesh_dst.exists():
            shutil.copy2(mesh_src, mesh_dst)
        sources["files"][mesh_name] = sha256_file(mesh_dst)

    snapshot_download(
        CLIP_ID,
        token=os.environ.get("HF_TOKEN"),
        allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"],
    )
    snapshot_download(
        VGGT_ID,
        token=os.environ.get("HF_TOKEN"),
        allow_patterns=["*.json", "*.pt", "*.pth", "*.safetensors"],
    )
    return sources


def description_from_record(record: dict[str, Any]) -> str:
    value = record["conversations"][0]["value"]
    return value.split("\n")[-1].strip()


@dataclass
class Example:
    scene: str
    uid: str
    text: str
    target: int


@dataclass
class SceneData:
    scene: str
    boxes: np.ndarray
    object_ids: list[str]
    examples: list[Example]
    image_paths: list[str]
    sensor_points: np.ndarray


def robust_normalize_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    good = np.isfinite(points).all(axis=1)
    points = points[good]
    if len(points) == 0:
        return np.zeros((1, 3), dtype=np.float32)
    lo = np.quantile(points, 0.02, axis=0)
    hi = np.quantile(points, 0.98, axis=0)
    scale = np.maximum(hi - lo, 1e-4)
    return np.clip((points - lo) / scale * 2.0 - 1.0, -2.0, 2.0)


def deterministic_sample(array: np.ndarray, count: int) -> np.ndarray:
    if len(array) == 0:
        return np.zeros((count, array.shape[-1]), dtype=np.float32)
    if len(array) >= count:
        idx = np.linspace(0, len(array) - 1, count).round().astype(np.int64)
        return array[idx].astype(np.float32)
    reps = math.ceil(count / len(array))
    return np.tile(array, (reps, 1))[:count].astype(np.float32)


def load_sensor_tokens(mesh_path: Path, count: int) -> np.ndarray:
    vertex = PlyData.read(str(mesh_path))["vertex"].data
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    xyz = robust_normalize_points(xyz)
    xyz = deterministic_sample(xyz, count)
    confidence = np.ones((count, 1), dtype=np.float32)
    zeros = np.zeros((count, 3), dtype=np.float32)
    return np.concatenate([xyz, confidence, zeros], axis=1)


def load_scene_data(root: Path, config: dict[str, Any]) -> dict[str, SceneData]:
    records = json.loads((root / "scanrefer_vg_val_llava_style.json").read_text())
    by_scene: dict[str, list[dict[str, Any]]] = {scene: [] for scene in ALL_SCENES}
    for record in records:
        scene = record["video"].split("/")[-1]
        if scene in by_scene:
            by_scene[scene].append(record)

    output: dict[str, SceneData] = {}
    for scene in ALL_SCENES:
        scene_records = by_scene[scene]
        if not scene_records:
            raise RuntimeError(f"No ScanRefer annotations found for required scene {scene}")
        object_to_box: dict[str, list[float]] = {}
        for record in scene_records:
            object_to_box.setdefault(str(record["metadata"]["object_id"]), record["box"])
        object_ids = sorted(object_to_box, key=lambda x: int(x))
        object_index = {obj: i for i, obj in enumerate(object_ids)}
        boxes = np.asarray([object_to_box[obj] for obj in object_ids], dtype=np.float32)

        selected: list[dict[str, Any]] = []
        per_object: dict[str, int] = {}
        for record in sorted(scene_records, key=lambda r: int(r["id"])):
            obj = str(record["metadata"]["object_id"])
            if per_object.get(obj, 0) >= 8:
                continue
            selected.append(record)
            per_object[obj] = per_object.get(obj, 0) + 1
        selected = selected[: int(config["max_examples_per_scene"])]
        examples = [
            Example(
                scene=scene,
                uid=f"{scene}:{record['id']}",
                text=description_from_record(record),
                target=object_index[str(record["metadata"]["object_id"])],
            )
            for record in selected
        ]

        frame_paths = [str(path) for path in sorted((root / scene).glob("*.jpg"))]
        frame_count = int(config["frames_per_scene"])
        indices = np.linspace(0, len(frame_paths) - 1, frame_count).round().astype(int)
        image_paths = [frame_paths[i] for i in indices]
        sensor_points = load_sensor_tokens(
            root / scene / f"{scene}_vh_clean_2.ply",
            int(config["tokens_per_stream"]),
        )
        output[scene] = SceneData(
            scene=scene,
            boxes=boxes,
            object_ids=object_ids,
            examples=examples,
            image_paths=image_paths,
            sensor_points=sensor_points,
        )
    if len({scene for scene in TEST_SCENES if output[scene].examples}) != len(TEST_SCENES):
        raise RuntimeError("Held-out split does not contain all required annotated scenes")
    return output


def pool_tokens(tokens: torch.Tensor, count: int) -> torch.Tensor:
    # [N, D] -> [count, D]
    return F.adaptive_avg_pool1d(tokens.T.unsqueeze(0), count).squeeze(0).T


def load_frozen_models(device: torch.device) -> tuple[CLIPModel, AutoProcessor, Any]:
    from vggt.models.vggt import VGGT

    clip = CLIPModel.from_pretrained(CLIP_ID).eval().to(device)
    processor = AutoProcessor.from_pretrained(CLIP_ID)
    vggt = VGGT.from_pretrained(VGGT_ID).eval().to(device)
    for model in (clip, vggt):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return clip, processor, vggt


def encode_texts(
    clip: CLIPModel,
    processor: AutoProcessor,
    texts: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for start in range(0, len(texts), 64):
        batch = texts[start : start + 64]
        inputs = processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            features = clip.get_text_features(**inputs)
            features = F.normalize(features.float(), dim=-1).cpu()
        result.update({text: feature for text, feature in zip(batch, features)})
    return result


def encode_scene(
    scene: SceneData,
    clip: CLIPModel,
    processor: AutoProcessor,
    vggt: Any,
    device: torch.device,
    token_count: int,
) -> dict[str, torch.Tensor]:
    from vggt.utils.load_fn import load_and_preprocess_images

    pil_images = [Image.open(path).convert("RGB") for path in scene.image_paths]
    clip_inputs = processor(images=pil_images, return_tensors="pt")
    pixel_values = clip_inputs["pixel_values"].to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        vision = clip.vision_model(pixel_values=pixel_values).last_hidden_state[:, 1:]
        rgb = clip.visual_projection(vision)
    rgb = pool_tokens(rgb.float().reshape(-1, rgb.shape[-1]), token_count).cpu()

    images = load_and_preprocess_images(scene.image_paths).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        aggregated_tokens_list, ps_idx = vggt.aggregator(images.unsqueeze(0))
        last = aggregated_tokens_list[-1][0]
        patch_tokens = last[:, int(ps_idx) :, :]
        point_map, point_conf = vggt.point_head(
            aggregated_tokens_list,
            images.unsqueeze(0),
            ps_idx,
        )
    implicit = pool_tokens(
        patch_tokens.float().reshape(-1, patch_tokens.shape[-1]),
        token_count,
    ).cpu()

    points = point_map[0].float().cpu().numpy().reshape(-1, 3)
    confidence = point_conf[0].float().cpu().numpy().reshape(-1, 1)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(confidence[:, 0])
    points = robust_normalize_points(points[finite])
    confidence = confidence[finite]
    if len(confidence):
        confidence = (confidence - np.quantile(confidence, 0.05)) / max(
            float(np.quantile(confidence, 0.95) - np.quantile(confidence, 0.05)),
            1e-6,
        )
        confidence = np.clip(confidence, 0.0, 1.0)
    spatial_count = len(points)
    frame_pos = np.linspace(-1.0, 1.0, spatial_count, dtype=np.float32)[:, None]
    uv = np.stack(
        [
            np.sin(np.linspace(0, 4 * np.pi, spatial_count)),
            np.cos(np.linspace(0, 4 * np.pi, spatial_count)),
        ],
        axis=1,
    ).astype(np.float32)
    reconstructed = np.concatenate([points, confidence, frame_pos, uv], axis=1)
    reconstructed = torch.from_numpy(deterministic_sample(reconstructed, token_count))
    sensor = torch.from_numpy(scene.sensor_points.copy())
    return {
        "rgb": rgb,
        "implicit": implicit,
        "reconstructed": reconstructed,
        "sensor": sensor,
    }


def normalize_boxes(boxes: np.ndarray) -> np.ndarray:
    centers = boxes[:, :3]
    sizes = boxes[:, 3:6]
    lo = np.min(centers - sizes / 2, axis=0)
    hi = np.max(centers + sizes / 2, axis=0)
    scale = np.maximum(hi - lo, 1e-4)
    norm_centers = (centers - lo) / scale * 2 - 1
    norm_sizes = sizes / scale
    return np.concatenate([norm_centers, norm_sizes], axis=1).astype(np.float32)


class GroundingDataset(Dataset):
    def __init__(
        self,
        examples: list[Example],
        scenes: dict[str, SceneData],
        features: dict[str, dict[str, torch.Tensor]],
        text_features: dict[str, torch.Tensor],
        max_candidates: int,
    ) -> None:
        self.examples = examples
        self.scenes = scenes
        self.features = features
        self.text_features = text_features
        self.max_candidates = max_candidates

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        scene = self.scenes[example.scene]
        count = len(scene.boxes)
        boxes = np.zeros((self.max_candidates, 6), dtype=np.float32)
        boxes[:count] = normalize_boxes(scene.boxes)
        raw_boxes = np.zeros((self.max_candidates, 6), dtype=np.float32)
        raw_boxes[:count] = scene.boxes
        mask = np.zeros(self.max_candidates, dtype=np.bool_)
        mask[:count] = True
        return {
            "uid": example.uid,
            "scene": example.scene,
            "text": self.text_features[example.text],
            "boxes": torch.from_numpy(boxes),
            "raw_boxes": torch.from_numpy(raw_boxes),
            "candidate_mask": torch.from_numpy(mask),
            "target": torch.tensor(example.target, dtype=torch.long),
            **self.features[example.scene],
        }


class MatchedAdapter(nn.Module):
    def __init__(self, implicit_dim: int, hidden: int, heads: int = 4) -> None:
        super().__init__()
        self.rgb_proj = nn.Linear(512, hidden)
        self.implicit_proj = nn.Linear(implicit_dim, hidden)
        self.explicit_proj = nn.Linear(7, hidden)
        self.text_proj = nn.Linear(512, hidden)
        self.box_proj = nn.Sequential(
            nn.Linear(6, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.iea = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.target_attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.concat_proj = nn.Linear(hidden * 2, hidden)
        self.rgb_norm = nn.LayerNorm(hidden)
        self.implicit_norm = nn.LayerNorm(hidden)
        self.explicit_norm = nn.LayerNorm(hidden)
        self.fusion_norm = nn.LayerNorm(hidden)
        self.query_norm = nn.LayerNorm(hidden)
        self.score = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.null_token = nn.Parameter(torch.zeros(1, 1, hidden))

    def forward(self, batch: dict[str, Any], variant: str) -> torch.Tensor:
        rgb = self.rgb_norm(self.rgb_proj(batch["rgb"]))
        implicit = self.implicit_norm(self.implicit_proj(batch["implicit"]))
        explicit_source = batch["sensor"] if variant == "sensor-iea" else batch["reconstructed"]
        explicit = self.explicit_norm(self.explicit_proj(explicit_source))

        if variant == "rgb":
            fused = rgb
        elif variant == "implicit":
            fused = rgb + implicit
        elif variant == "explicit-recon":
            fused = rgb + explicit
        elif variant in {"combined-iea", "sensor-iea", "shuffle-explicit"}:
            if variant == "shuffle-explicit":
                explicit = torch.roll(explicit, shifts=1, dims=0)
            exchanged, _ = self.iea(implicit, explicit, explicit, need_weights=False)
            fused = rgb + self.fusion_norm(implicit + exchanged)
        elif variant == "concat":
            fused = rgb + self.fusion_norm(self.concat_proj(torch.cat([implicit, explicit], dim=-1)))
        elif variant == "addition":
            fused = rgb + self.fusion_norm(implicit + explicit)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        query = self.query_norm(
            self.text_proj(batch["text"]).unsqueeze(1) + self.box_proj(batch["boxes"])
        )
        attended, _ = self.target_attention(query, fused, fused, need_weights=False)
        logits = self.score(query + attended).squeeze(-1)
        return logits.masked_fill(~batch["candidate_mask"], -1e9)


def aabb_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    a_min, a_max = box_a[:3] - box_a[3:] / 2, box_a[:3] + box_a[3:] / 2
    b_min, b_max = box_b[:3] - box_b[3:] / 2, box_b[:3] + box_b[3:] / 2
    inter = torch.clamp(torch.minimum(a_max, b_max) - torch.maximum(a_min, b_min), min=0)
    inter_volume = float(torch.prod(inter))
    a_volume = float(torch.prod(torch.clamp(box_a[3:], min=0)))
    b_volume = float(torch.prod(torch.clamp(box_b[3:], min=0)))
    return inter_volume / max(a_volume + b_volume - inter_volume, 1e-9)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def evaluate(
    model: MatchedAdapter,
    loader: DataLoader,
    device: torch.device,
    variant: str,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            device_batch = move_batch(batch, device)
            logits = model(device_batch, variant)
            pred = logits.argmax(dim=1).cpu()
            target = batch["target"]
            for i in range(len(pred)):
                iou = aabb_iou(batch["raw_boxes"][i, pred[i]], batch["raw_boxes"][i, target[i]])
                rows.append(
                    {
                        "uid": batch["uid"][i],
                        "scene": batch["scene"][i],
                        "pred": int(pred[i]),
                        "target": int(target[i]),
                        "iou": iou,
                        "acc25": float(iou >= 0.25),
                        "acc50": float(iou >= 0.50),
                        "exact": float(pred[i] == target[i]),
                    }
                )
    elapsed = time.perf_counter() - start
    scene_metrics: dict[str, dict[str, float]] = {}
    for scene in sorted({row["scene"] for row in rows}):
        group = [row for row in rows if row["scene"] == scene]
        scene_metrics[scene] = {
            "n": len(group),
            "acc25": float(np.mean([row["acc25"] for row in group])),
            "acc50": float(np.mean([row["acc50"] for row in group])),
            "exact": float(np.mean([row["exact"] for row in group])),
        }
    return {
        "n": len(rows),
        "acc25": float(np.mean([row["acc25"] for row in rows])),
        "acc50": float(np.mean([row["acc50"] for row in rows])),
        "exact": float(np.mean([row["exact"] for row in rows])),
        "scene_metrics": scene_metrics,
        "throughput_examples_s": len(rows) / max(elapsed, 1e-9),
        "predictions": rows,
    }


def seed_worker(
    rank: int,
    seed: int,
    config: dict[str, Any],
    data_root: str,
    result_path: str,
) -> None:
    started = time.perf_counter()
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True

        scenes = load_scene_data(Path(data_root), config)
        clip, processor, vggt = load_frozen_models(device)
        feature_start = time.perf_counter()
        scene_features = {
            scene: encode_scene(
                data,
                clip,
                processor,
                vggt,
                device,
                int(config["tokens_per_stream"]),
            )
            for scene, data in scenes.items()
        }
        all_texts = sorted({ex.text for scene in scenes.values() for ex in scene.examples})
        text_features = encode_texts(clip, processor, all_texts, device)
        feature_elapsed = time.perf_counter() - feature_start
        del clip, vggt
        torch.cuda.empty_cache()

        train_examples = [ex for scene in TRAIN_SCENES for ex in scenes[scene].examples]
        internal_val = [
            ex
            for ex in train_examples
            if int(hashlib.sha256(ex.uid.encode()).hexdigest(), 16) % 5 == 0
        ]
        train_examples = [ex for ex in train_examples if ex not in internal_val]
        test_examples = [ex for scene in TEST_SCENES for ex in scenes[scene].examples]
        max_candidates = max(len(scene.boxes) for scene in scenes.values())

        def make_loader(examples: list[Example], shuffle: bool, batch_size: int) -> DataLoader:
            generator = torch.Generator().manual_seed(seed)
            return DataLoader(
                GroundingDataset(
                    examples,
                    scenes,
                    scene_features,
                    text_features,
                    max_candidates,
                ),
                batch_size=batch_size,
                shuffle=shuffle,
                generator=generator,
                num_workers=0,
                pin_memory=True,
            )

        train_loader = make_loader(train_examples, True, int(config["batch_size"]))
        val_loader = make_loader(internal_val, False, int(config["batch_size"]) * 2)
        test_loader = make_loader(test_examples, False, int(config["batch_size"]) * 2)
        implicit_dim = int(scene_features[ALL_SCENES[0]]["implicit"].shape[-1])
        model = MatchedAdapter(implicit_dim, int(config["hidden_dim"])).to(device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["epochs"]) * len(train_loader),
        )
        best_state = None
        best_val = -1.0
        curve: list[dict[str, float]] = []
        train_seen = 0
        train_start = time.perf_counter()
        for epoch in range(int(config["epochs"])):
            model.train()
            losses = []
            for batch in train_loader:
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch, str(config["variant"]))
                loss = F.cross_entropy(logits, batch["target"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach()))
                train_seen += len(batch["target"])
            if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == int(config["epochs"]):
                val = evaluate(model, val_loader, device, str(config["variant"]))
                curve.append(
                    {
                        "epoch": epoch + 1,
                        "loss": float(np.mean(losses)),
                        "val_acc25": val["acc25"],
                        "val_acc50": val["acc50"],
                    }
                )
                if val["acc25"] > best_val:
                    best_val = val["acc25"]
                    best_state = copy.deepcopy(model.state_dict())
        train_elapsed = time.perf_counter() - train_start
        if best_state is not None:
            model.load_state_dict(best_state)
        test = evaluate(model, test_loader, device, str(config["variant"]))
        result = {
            "status": "ok",
            "paper_id": PAPER_ID,
            "variant": config["variant"],
            "label": config["label"],
            "seed": seed,
            "gpu": torch.cuda.get_device_name(0),
            "trainable_parameters": trainable_params,
            "train_examples": len(train_examples),
            "internal_val_examples": len(internal_val),
            "heldout_examples": len(test_examples),
            "train_scenes": TRAIN_SCENES,
            "heldout_scenes": TEST_SCENES,
            "candidate_counts": {scene: len(data.boxes) for scene, data in scenes.items()},
            "feature_elapsed_s": feature_elapsed,
            "feature_scenes_s": len(scenes) / max(feature_elapsed, 1e-9),
            "train_elapsed_s": train_elapsed,
            "train_examples_s": train_seen / max(train_elapsed, 1e-9),
            "curve": curve,
            "test": test,
            "elapsed_s": time.perf_counter() - started,
        }
        Path(result_path).write_text(json.dumps(result))
    except Exception as exc:
        Path(result_path).write_text(
            json.dumps(
                {
                    "status": "failed",
                    "seed": seed,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        )


def aggregate_results(results: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
    successful = [result for result in results if result.get("status") == "ok"]
    if not successful:
        return {"status": "failed", "results": results, "wall_s": wall_s}
    summary = {
        "status": "ok" if len(successful) == len(results) else "partial",
        "variant": successful[0]["variant"],
        "label": successful[0]["label"],
        "seeds_requested": [result.get("seed") for result in results],
        "seeds_successful": [result["seed"] for result in successful],
        "trainable_parameters": successful[0]["trainable_parameters"],
        "train_examples": successful[0]["train_examples"],
        "internal_val_examples": successful[0]["internal_val_examples"],
        "heldout_examples": successful[0]["heldout_examples"],
        "acc25_mean": float(np.mean([result["test"]["acc25"] for result in successful])),
        "acc25_std": float(np.std([result["test"]["acc25"] for result in successful], ddof=1))
        if len(successful) > 1
        else 0.0,
        "acc50_mean": float(np.mean([result["test"]["acc50"] for result in successful])),
        "acc50_std": float(np.std([result["test"]["acc50"] for result in successful], ddof=1))
        if len(successful) > 1
        else 0.0,
        "exact_mean": float(np.mean([result["test"]["exact"] for result in successful])),
        "train_examples_s_mean": float(
            np.mean([result["train_examples_s"] for result in successful])
        ),
        "inference_examples_s_mean": float(
            np.mean([result["test"]["throughput_examples_s"] for result in successful])
        ),
        "feature_scenes_s_mean": float(
            np.mean([result["feature_scenes_s"] for result in successful])
        ),
        "gpu_model": successful[0]["gpu"],
        "allocated_gpu_count": len(results),
        "wall_s": wall_s,
        "per_seed": successful,
        "failures": [result for result in results if result.get("status") != "ok"],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    wall_start = time.perf_counter()
    available_gpus = torch.cuda.device_count()
    seeds = list(config["seeds"])
    if available_gpus < len(seeds):
        raise RuntimeError(f"Need {len(seeds)} GPUs, found {available_gpus}")
    emit(
        "run_start",
        started_utc=started_utc,
        paper_id=PAPER_ID,
        config=config,
        backend="kubernetes",
        gpu_model_expected=GPU_MODEL,
        visible_gpu_count=available_gpus,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
    )
    data_root = Path("/tmp/vlm_ie3d_scanrefer_reproduction")
    sources = prepare_assets(data_root)
    emit(
        "dataset_ready",
        sources=sources,
        train_scenes=TRAIN_SCENES,
        heldout_scenes=TEST_SCENES,
        split_policy="held out by base ScanNet scene group",
        primary_task="proposal-refined ScanRefer-style 3D grounding",
        primary_metrics=["Acc@0.25", "Acc@0.50"],
    )

    result_dir = Path("/tmp/vlm_ie3d_seed_results")
    result_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    processes = []
    for rank, seed in enumerate(seeds):
        result_path = result_dir / f"seed_{seed}.json"
        if result_path.exists():
            result_path.unlink()
        process = ctx.Process(
            target=seed_worker,
            args=(rank, int(seed), config, str(data_root), str(result_path)),
        )
        process.start()
        processes.append((process, result_path))
    for process, _ in processes:
        process.join()

    results = []
    for process, path in processes:
        if path.exists():
            results.append(json.loads(path.read_text()))
        else:
            results.append(
                {
                    "status": "failed",
                    "seed": None,
                    "error": f"worker exit code {process.exitcode}, no result",
                }
            )
    wall_s = time.perf_counter() - wall_start
    summary = aggregate_results(results, wall_s)
    emit("seed_results", results=results)
    emit(
        "terminal_summary",
        **summary,
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        evidence_note=(
            "Acc@0.25/0.50 use the selected candidate's axis-aligned 3D box. "
            "This is a bounded proposal-refined grounding test, not the paper's full VLM training."
        ),
    )
    if summary["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
