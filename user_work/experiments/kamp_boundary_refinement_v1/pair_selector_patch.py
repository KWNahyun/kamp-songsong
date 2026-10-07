"""Select between the original and refined box of the same frozen query."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.zoo.dfine.dfine_decoder import DFINETransformer
from boundary_patch import cxcywh_to_xyxy


class BaseRefinedPairSelector(nn.Module):
    """Shared relative-quality head for an original/refined query pair."""

    def __init__(self, hidden_dim: int, reg_max: int) -> None:
        super().__init__()
        self.reg_max = reg_max
        # feature + FDR top-4 stats + score + geometry + applied edge delta
        input_dim = hidden_dim + 4 * 5 + 1 + 6 + 4
        self.individual = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 160),
            nn.SiLU(),
            nn.Linear(160, 128),
            nn.SiLU(),
        )
        self.relative_head = nn.Sequential(
            nn.LayerNorm(128 * 3),
            nn.Linear(128 * 3, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.relative_head[-1].weight)
        nn.init.zeros_(self.relative_head[-1].bias)

    def _embed(
        self,
        shared: torch.Tensor,
        boxes: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        size = boxes[..., 2:].clamp_min(1e-6)
        geometry = torch.cat((boxes, size.log()), dim=-1)
        return self.individual(torch.cat((shared, geometry, delta), dim=-1))

    def forward(
        self,
        features: torch.Tensor,
        corners: torch.Tensor,
        base_logits: torch.Tensor,
        base_boxes: torch.Tensor,
        refined_boxes: torch.Tensor,
        refined_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, queries, _ = corners.shape
        probability = corners.reshape(batch, queries, 4, self.reg_max + 1).softmax(dim=-1)
        top4 = probability.topk(4, dim=-1).values
        distribution = torch.cat((top4, top4.mean(dim=-1, keepdim=True)), dim=-1).flatten(2)
        score = base_logits.sigmoid().amax(dim=-1, keepdim=True)
        shared = torch.cat((features, distribution, score), dim=-1)
        base_embedding = self._embed(shared, base_boxes, torch.zeros_like(refined_delta))
        refined_embedding = self._embed(shared, refined_boxes, refined_delta)
        base_input = torch.cat(
            (base_embedding, refined_embedding, base_embedding - refined_embedding), dim=-1
        )
        refined_input = torch.cat(
            (refined_embedding, base_embedding, refined_embedding - base_embedding), dim=-1
        )
        return self.relative_head(base_input), self.relative_head(refined_input)


def _combine_logits(base_logits: torch.Tensor, quality_logits: torch.Tensor) -> torch.Tensor:
    probability = torch.sqrt(
        (base_logits.sigmoid() * quality_logits.sigmoid()).clamp_min(1e-8)
    ).clamp(1e-6, 1 - 1e-6)
    return torch.logit(probability)


def install_pair_selector() -> None:
    if not getattr(DFINETransformer, "_kamp_boundary_refinement", False):
        raise RuntimeError("Install the box boundary refiner first")
    if getattr(DFINETransformer, "_kamp_pair_selector", False):
        raise RuntimeError("Install pair selector only once per process")
    original_init = DFINETransformer.__init__
    original_forward = DFINETransformer.forward

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(85231)
            self.kamp_pair_selector = BaseRefinedPairSelector(self.hidden_dim, self.reg_max)

    def patched_forward(self, feats, targets=None):
        result = original_forward(self, feats, targets)
        features = self.decoder.layers[self.eval_idx]._kamp_boundary_output
        corners = self.decoder.lqe_layers[self.eval_idx]._kamp_boundary_corners
        features = features[:, -self.num_queries :].detach()
        corners = corners[:, -self.num_queries :].detach()
        base_boxes = result["pred_boxes_base"].detach()
        refined_boxes = result["pred_boxes"].detach()
        base_logits = result["pred_logits"]
        base_quality, refined_quality = self.kamp_pair_selector(
            features,
            corners,
            base_logits.detach(),
            base_boxes,
            refined_boxes,
            result["pred_boundary_delta"].detach(),
        )
        result["pred_boxes_refined"] = refined_boxes
        result["pred_pair_quality_base"] = base_quality
        result["pred_pair_quality_refined"] = refined_quality
        choose_refined = refined_quality > base_quality
        selected_boxes = torch.where(choose_refined.expand_as(base_boxes), refined_boxes, base_boxes)
        selected_quality = torch.maximum(base_quality, refined_quality)
        result["pred_boxes"] = selected_boxes
        result["pred_logits_base"] = base_logits
        result["pred_logits"] = _combine_logits(base_logits, selected_quality)
        return result

    DFINETransformer.__init__ = patched_init
    DFINETransformer.forward = patched_forward
    DFINETransformer._kamp_pair_selector = True


def max_iou_quality(boxes: torch.Tensor, targets: list[dict]) -> torch.Tensor:
    result = []
    predicted = cxcywh_to_xyxy(boxes)
    for image_boxes, target in zip(predicted, targets):
        truth = cxcywh_to_xyxy(target["boxes"])
        if len(truth) == 0:
            result.append(torch.zeros(len(image_boxes), device=boxes.device, dtype=boxes.dtype))
            continue
        left_top = torch.maximum(image_boxes[:, None, :2], truth[None, :, :2])
        right_bottom = torch.minimum(image_boxes[:, None, 2:], truth[None, :, 2:])
        intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
        pred_area = (image_boxes[:, 2:] - image_boxes[:, :2]).clamp_min(0).prod(dim=-1)
        truth_area = (truth[:, 2:] - truth[:, :2]).clamp_min(0).prod(dim=-1)
        iou = intersection / (
            pred_area[:, None] + truth_area[None, :] - intersection
        ).clamp_min(1e-8)
        result.append(iou.max(dim=1).values)
    return torch.stack(result)


def pair_quality_loss(outputs: dict, targets: list[dict], beta: float = 2.0) -> torch.Tensor:
    base_target = max_iou_quality(outputs["pred_boxes_base"], targets)
    refined_target = max_iou_quality(outputs["pred_boxes_refined"], targets)
    targets_stacked = torch.stack((base_target, refined_target), dim=-1)
    logits = torch.cat(
        (outputs["pred_pair_quality_base"], outputs["pred_pair_quality_refined"]), dim=-1
    )
    modulation = (targets_stacked - logits.sigmoid()).abs().pow(beta)
    return (
        F.binary_cross_entropy_with_logits(logits, targets_stacked, reduction="none")
        * modulation
    ).mean()

