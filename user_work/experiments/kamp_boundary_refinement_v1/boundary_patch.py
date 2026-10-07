"""Frozen-candidate boundary refinement heads for D-FINE.

Two heads share the same four-edge residual target:

* ``box`` uses one global query representation and predicts all four edges.
* ``edge`` processes each FDR edge distribution separately with a shared edge
  predictor.  This is the architecture hypothesis under test.

All detector inputs are detached.  The base detector can therefore be frozen
and audited independently from the refiner.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.zoo.dfine.dfine_decoder import DFINETransformer


MAX_EDGE_SHIFT = 0.75


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    center = (boxes[..., :2] + boxes[..., 2:]) / 2
    size = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1e-5)
    return torch.cat((center, size), dim=-1)


def pairwise_iou_xyxy(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    left_top = torch.maximum(first[:, :, None, :2], second[:, None, :, :2])
    right_bottom = torch.minimum(first[:, :, None, 2:], second[:, None, :, 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
    first_area = (first[..., 2:] - first[..., :2]).clamp_min(0).prod(dim=-1)
    second_area = (second[..., 2:] - second[..., :2]).clamp_min(0).prod(dim=-1)
    union = first_area[:, :, None] + second_area[:, None, :] - intersection
    return intersection / union.clamp_min(1e-8)


def apply_edge_delta(boxes: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    valid = (boxes[..., 2] > 1e-5) & (boxes[..., 3] > 1e-5)
    edges = cxcywh_to_xyxy(boxes)
    width = boxes[..., 2].clamp_min(1e-5)
    height = boxes[..., 3].clamp_min(1e-5)
    scale = torch.stack((width, height, width, height), dim=-1)
    moved = edges + delta * scale
    # The target residual always describes a valid GT box.  This guard only
    # prevents an early-training outlier from creating a negative-size box.
    x1, y1, x2, y2 = moved.unbind(dim=-1)
    x2 = torch.maximum(x2, x1 + 1e-5)
    y2 = torch.maximum(y2, y1 + 1e-5)
    refined = xyxy_to_cxcywh(torch.stack((x1, y1, x2, y2), dim=-1))
    return torch.where(valid.unsqueeze(-1), refined, boxes)


class GlobalBoxRefiner(nn.Module):
    """Generic four-edge residual baseline from aggregated query statistics."""

    def __init__(self, hidden_dim: int, reg_max: int) -> None:
        super().__init__()
        self.reg_max = reg_max
        input_dim = hidden_dim + 4 * 5 + 1 + 6
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 4),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        corners: torch.Tensor,
        boxes: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        batch, queries, _ = corners.shape
        probability = corners.reshape(batch, queries, 4, self.reg_max + 1).softmax(dim=-1)
        top4 = probability.topk(4, dim=-1).values
        distribution = torch.cat((top4, top4.mean(dim=-1, keepdim=True)), dim=-1).flatten(2)
        score = base_logits.sigmoid().amax(dim=-1, keepdim=True)
        size = boxes[..., 2:].clamp_min(1e-6)
        geometry = torch.cat((boxes, size.log()), dim=-1)
        raw = self.network(torch.cat((features, distribution, score, geometry), dim=-1))
        return MAX_EDGE_SHIFT * raw.tanh()


class EdgeAwareRefiner(nn.Module):
    """Shared per-edge predictor conditioned on each full FDR distribution."""

    def __init__(self, hidden_dim: int, reg_max: int) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.global_embedding = nn.Sequential(
            nn.LayerNorm(hidden_dim + 1 + 6),
            nn.Linear(hidden_dim + 1 + 6, 128),
            nn.SiLU(),
            nn.Linear(128, 96),
            nn.SiLU(),
        )
        self.distribution_embedding = nn.Sequential(
            nn.LayerNorm(reg_max + 1 + 3),
            nn.Linear(reg_max + 1 + 3, 64),
            nn.SiLU(),
            nn.Linear(64, 48),
            nn.SiLU(),
        )
        self.edge_embedding = nn.Parameter(torch.zeros(4, 16))
        nn.init.normal_(self.edge_embedding, std=0.02)
        self.shared_edge_head = nn.Sequential(
            nn.LayerNorm(96 + 48 + 16),
            nn.Linear(96 + 48 + 16, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.shared_edge_head[-1].weight)
        nn.init.zeros_(self.shared_edge_head[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        corners: torch.Tensor,
        boxes: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        batch, queries, _ = corners.shape
        probability = corners.reshape(batch, queries, 4, self.reg_max + 1).softmax(dim=-1)
        grid = torch.linspace(0, 1, self.reg_max + 1, device=corners.device, dtype=corners.dtype)
        mean = (probability * grid).sum(dim=-1, keepdim=True)
        variance = (probability * (grid - mean).square()).sum(dim=-1, keepdim=True)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        entropy = entropy / torch.log(torch.tensor(self.reg_max + 1.0, device=corners.device))
        edge_distribution = torch.cat((probability, mean, variance.sqrt(), entropy), dim=-1)

        score = base_logits.sigmoid().amax(dim=-1, keepdim=True)
        size = boxes[..., 2:].clamp_min(1e-6)
        geometry = torch.cat((boxes, size.log()), dim=-1)
        global_embedding = self.global_embedding(torch.cat((features, score, geometry), dim=-1))
        global_embedding = global_embedding.unsqueeze(2).expand(-1, -1, 4, -1)
        distribution_embedding = self.distribution_embedding(edge_distribution)
        edge_identity = self.edge_embedding.view(1, 1, 4, -1).expand(batch, queries, -1, -1)
        raw = self.shared_edge_head(
            torch.cat((global_embedding, distribution_embedding, edge_identity), dim=-1)
        ).squeeze(-1)
        return MAX_EDGE_SHIFT * raw.tanh()


def _capture_layer_output(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
    module._kamp_boundary_output = output


def _capture_lqe_input(module: nn.Module, inputs: tuple) -> None:
    module._kamp_boundary_corners = inputs[1]


def install_model(mode: str) -> None:
    if mode not in {"box", "edge"}:
        raise ValueError(mode)
    if getattr(DFINETransformer, "_kamp_boundary_refinement", False):
        raise RuntimeError("Install boundary refinement only once per process")
    original_init = DFINETransformer.__init__
    original_forward = DFINETransformer.forward

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(74123)
            refiner_class = GlobalBoxRefiner if mode == "box" else EdgeAwareRefiner
            self.kamp_boundary_refiner = refiner_class(self.hidden_dim, self.reg_max)
        self.decoder.layers[self.eval_idx].register_forward_hook(_capture_layer_output)
        self.decoder.lqe_layers[self.eval_idx].register_forward_pre_hook(_capture_lqe_input)

    def patched_forward(self, feats, targets=None):
        result = original_forward(self, feats, targets)
        features = self.decoder.layers[self.eval_idx]._kamp_boundary_output
        corners = self.decoder.lqe_layers[self.eval_idx]._kamp_boundary_corners
        features = features[:, -self.num_queries :].detach()
        corners = corners[:, -self.num_queries :].detach()
        base_boxes = result["pred_boxes"]
        base_logits = result["pred_logits"]
        delta = self.kamp_boundary_refiner(
            features,
            corners,
            base_boxes.detach(),
            base_logits.detach(),
        )
        result["pred_boxes_base"] = base_boxes
        result["pred_boundary_delta"] = delta
        result["pred_boxes"] = apply_edge_delta(base_boxes.detach(), delta)
        return result

    DFINETransformer.__init__ = patched_init
    DFINETransformer.forward = patched_forward
    DFINETransformer._kamp_boundary_refinement = True


def assign_topk_targets(
    base_boxes: torch.Tensor,
    targets: list[dict],
    topk_per_gt: int = 10,
    minimum_iou: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign each selected query to its highest-IoU selected GT."""
    batch_size, queries, _ = base_boxes.shape
    assigned = torch.zeros_like(base_boxes)
    positive = torch.zeros(batch_size, queries, dtype=torch.bool, device=base_boxes.device)
    assigned_iou = torch.zeros(batch_size, queries, dtype=base_boxes.dtype, device=base_boxes.device)
    base_xyxy = cxcywh_to_xyxy(base_boxes)
    for batch_index, target in enumerate(targets):
        truth = target["boxes"]
        if len(truth) == 0:
            continue
        truth_xyxy = cxcywh_to_xyxy(truth)
        iou = pairwise_iou_xyxy(base_xyxy[batch_index : batch_index + 1], truth_xyxy[None])[0]
        valid = (base_boxes[batch_index, :, 2] > 1e-5) & (base_boxes[batch_index, :, 3] > 1e-5)
        iou = iou.masked_fill(~valid[:, None], 0)
        selected = torch.zeros_like(iou, dtype=torch.bool)
        k = min(topk_per_gt, queries)
        values, indices = iou.topk(k, dim=0)
        selected.scatter_(0, indices, values >= minimum_iou)
        eligible_iou = iou.masked_fill(~selected, -1)
        best_iou, best_gt = eligible_iou.max(dim=1)
        mask = best_iou >= minimum_iou
        positive[batch_index] = mask
        assigned[batch_index, mask] = truth[best_gt[mask]]
        assigned_iou[batch_index, mask] = best_iou[mask]
    return assigned, positive, assigned_iou


def generalized_iou_aligned(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    left_top = torch.maximum(first[:, :2], second[:, :2])
    right_bottom = torch.minimum(first[:, 2:], second[:, 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
    first_area = (first[:, 2:] - first[:, :2]).clamp_min(0).prod(dim=-1)
    second_area = (second[:, 2:] - second[:, :2]).clamp_min(0).prod(dim=-1)
    union = first_area + second_area - intersection
    iou = intersection / union.clamp_min(1e-8)
    enclosure_left_top = torch.minimum(first[:, :2], second[:, :2])
    enclosure_right_bottom = torch.maximum(first[:, 2:], second[:, 2:])
    enclosure = (enclosure_right_bottom - enclosure_left_top).clamp_min(0).prod(dim=-1)
    return iou - (enclosure - union) / enclosure.clamp_min(1e-8)


def refinement_loss(
    outputs: dict,
    targets: list[dict],
) -> tuple[torch.Tensor, dict[str, float]]:
    base_boxes = outputs["pred_boxes_base"].detach()
    assigned, positive, assigned_iou = assign_topk_targets(base_boxes, targets)
    if not positive.any():
        zero = outputs["pred_boundary_delta"].sum() * 0
        return zero, {"positive_queries": 0, "mean_base_iou": 0.0}

    base_edges = cxcywh_to_xyxy(base_boxes[positive])
    target_edges = cxcywh_to_xyxy(assigned[positive])
    base_size = base_boxes[positive][:, 2:].clamp_min(1e-5)
    scale = torch.stack(
        (base_size[:, 0], base_size[:, 1], base_size[:, 0], base_size[:, 1]), dim=-1
    )
    target_delta = ((target_edges - base_edges) / scale).clamp(-MAX_EDGE_SHIFT, MAX_EDGE_SHIFT)
    predicted_delta = outputs["pred_boundary_delta"][positive]
    residual_loss = F.smooth_l1_loss(predicted_delta, target_delta, beta=0.1)

    refined_edges = cxcywh_to_xyxy(outputs["pred_boxes"][positive])
    giou_loss = (1 - generalized_iou_aligned(refined_edges, target_edges)).mean()
    loss = residual_loss + 0.5 * giou_loss
    diagnostics = {
        "positive_queries": int(positive.sum()),
        "mean_base_iou": float(assigned_iou[positive].mean()),
        "loss_residual": float(residual_loss.detach()),
        "loss_giou": float(giou_loss.detach()),
    }
    return loss, diagnostics
