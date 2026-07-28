import torch

from src.reproduce import aabb_iou


def test_iou_identity_and_disjoint():
    box = torch.tensor([0.0, 0.0, 0.0, 2.0, 2.0, 2.0])
    assert aabb_iou(box, box) == 1.0
    other = torch.tensor([5.0, 0.0, 0.0, 2.0, 2.0, 2.0])
    assert aabb_iou(box, other) == 0.0
