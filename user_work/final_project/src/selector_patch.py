"""Detached candidate-quality heads for the first relative-selector pilot.

The detector keeps its original MAL training path.  The selector receives
detached final-query tensors, so its loss cannot change the backbone, encoder,
decoder, box heads, or original score heads.  This makes the unary-versus-
relation comparison a clean test of candidate selection rather than a hidden
detector retuning experiment.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.zoo.dfine.dfine_criterion import DFINECriterion
from src.zoo.dfine.dfine_decoder import DFINETransformer


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def _pairwise_iou(boxes: torch.Tensor) -> torch.Tensor:
    boxes = _cxcywh_to_xyxy(boxes)
    left_top = torch.maximum(boxes[:, :, None, :2], boxes[:, None, :, :2])
    right_bottom = torch.minimum(boxes[:, :, None, 2:], boxes[:, None, :, 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
    area = (boxes[..., 2:] - boxes[..., :2]).clamp_min(0).prod(dim=-1)
    union = area[:, :, None] + area[:, None, :] - intersection
    return intersection / union.clamp_min(1e-8)


def _max_iou_target(boxes: torch.Tensor, targets: list[dict]) -> torch.Tensor:
    """Maximum IoU to any labeled object for every regular query."""
    result = []
    pred_xyxy = _cxcywh_to_xyxy(boxes)
    for predicted, target in zip(pred_xyxy, targets):
        if len(target["boxes"]) == 0:
            result.append(torch.zeros(len(predicted), device=boxes.device, dtype=boxes.dtype))
            continue
        truth = _cxcywh_to_xyxy(target["boxes"])
        left_top = torch.maximum(predicted[:, None, :2], truth[None, :, :2])
        right_bottom = torch.minimum(predicted[:, None, 2:], truth[None, :, 2:])
        intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
        pred_area = (predicted[:, 2:] - predicted[:, :2]).clamp_min(0).prod(dim=-1)
        truth_area = (truth[:, 2:] - truth[:, :2]).clamp_min(0).prod(dim=-1)
        iou = intersection / (
            pred_area[:, None] + truth_area[None, :] - intersection
        ).clamp_min(1e-8)
        result.append(iou.max(dim=1).values)
    return torch.stack(result)


class RelativeQualitySelector(nn.Module):
    """Predict localization quality from one query or its overlapping peers."""

    def __init__(
        self,
        mode: str,
        hidden_dim: int,
        reg_max: int,
        topk_neighbors: int = 8,
    ) -> None:
        super().__init__()
        if mode not in {"unary", "relation"}:
            raise ValueError(f"Unknown selector mode: {mode}")
        self.mode = mode
        self.reg_max = reg_max
        self.topk_neighbors = topk_neighbors
        # Query feature + four-edge top-4 distribution statistics + base score
        # + cx/cy/w/h and log(w/h).
        input_dim = hidden_dim + 4 * 5 + 1 + 6
        embedding_dim = 128
        self.embedding = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
        )
        head_dim = embedding_dim if mode == "unary" else 3 * embedding_dim + 3
        self.quality_head = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        # q=0.5 at initialization.  The combined score is then a monotonic
        # transform of the base score, so initial candidate ordering is intact.
        nn.init.zeros_(self.quality_head[-1].weight)
        nn.init.zeros_(self.quality_head[-1].bias)

    def _distribution_statistics(self, corners: torch.Tensor) -> torch.Tensor:
        batch, queries, _ = corners.shape
        probability = corners.reshape(batch, queries, 4, self.reg_max + 1).softmax(dim=-1)
        top4 = probability.topk(4, dim=-1).values
        return torch.cat((top4, top4.mean(dim=-1, keepdim=True)), dim=-1).flatten(2)

    def forward(
        self,
        features: torch.Tensor,
        corners: torch.Tensor,
        boxes: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        base_score = base_logits.sigmoid().amax(dim=-1, keepdim=True)
        safe_size = boxes[..., 2:].clamp_min(1e-6)
        geometry = torch.cat((boxes, safe_size.log()), dim=-1)
        inputs = torch.cat(
            (features, self._distribution_statistics(corners), base_score, geometry), dim=-1
        )
        embedding = self.embedding(inputs)
        if self.mode == "unary":
            return self.quality_head(embedding)

        overlap = _pairwise_iou(boxes)
        count = overlap.shape[-1]
        diagonal = torch.eye(count, device=overlap.device, dtype=torch.bool).unsqueeze(0)
        overlap = overlap.masked_fill(diagonal, 0)
        k = min(self.topk_neighbors, max(count - 1, 1))
        neighbor_iou, neighbor_index = overlap.topk(k, dim=-1)
        batch_index = torch.arange(len(boxes), device=boxes.device)[:, None, None]
        neighbor_embedding = embedding[batch_index, neighbor_index]
        neighbor_score = base_score.squeeze(-1)[batch_index, neighbor_index]
        raw_weight = neighbor_iou * neighbor_score
        weight = raw_weight / raw_weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        context = (neighbor_embedding * weight.unsqueeze(-1)).sum(dim=2)
        relation_summary = torch.stack(
            (
                neighbor_iou.max(dim=-1).values,
                (neighbor_score * weight).sum(dim=-1),
                (neighbor_iou >= 0.1).to(boxes.dtype).mean(dim=-1),
            ),
            dim=-1,
        )
        relation = torch.cat(
            (embedding, context, embedding - context, relation_summary), dim=-1
        )
        return self.quality_head(relation)


def _capture_layer_output(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
    module._kamp_selector_output = output


def _capture_lqe_input(module: nn.Module, inputs: tuple) -> None:
    module._kamp_selector_corners = inputs[1]


def _combine_logits(base_logits: torch.Tensor, quality_logits: torch.Tensor) -> torch.Tensor:
    # Fixed geometric mean: neither branch can dominate by scale alone.
    base_probability = base_logits.sigmoid()
    quality_probability = quality_logits.sigmoid()
    combined = torch.sqrt((base_probability * quality_probability).clamp_min(1e-8))
    combined = combined.clamp(1e-6, 1 - 1e-6)
    return torch.logit(combined)


def install_model(mode: str) -> None:
    if getattr(DFINETransformer, "_kamp_relative_selector", False):
        raise RuntimeError("Install relative selector only once per process")
    original_init = DFINETransformer.__init__
    original_forward = DFINETransformer.forward

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Do not consume the experiment seed's RNG stream.  This preserves the
        # MAL detector initialization/data-order path used by the control run.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(91027)
            self.kamp_selector = RelativeQualitySelector(mode, self.hidden_dim, self.reg_max)
        self.decoder.layers[self.eval_idx].register_forward_hook(_capture_layer_output)
        self.decoder.lqe_layers[self.eval_idx].register_forward_pre_hook(_capture_lqe_input)

    def patched_forward(self, feats, targets=None):
        result = original_forward(self, feats, targets)
        query_features = self.decoder.layers[self.eval_idx]._kamp_selector_output
        corners = self.decoder.lqe_layers[self.eval_idx]._kamp_selector_corners
        # Denoising queries are prepended; regular queries are always the tail.
        query_features = query_features[:, -self.num_queries :].detach()
        corners = corners[:, -self.num_queries :].detach()
        boxes = result["pred_boxes"].detach()
        base_logits = result["pred_logits"]
        quality_logits = self.kamp_selector(
            query_features,
            corners,
            boxes,
            base_logits.detach(),
        )
        result["pred_quality_logits"] = quality_logits
        result["pred_logits_base"] = base_logits
        if not self.training:
            result["pred_logits"] = _combine_logits(base_logits, quality_logits)
        return result

    DFINETransformer.__init__ = patched_init
    DFINETransformer.forward = patched_forward
    DFINETransformer._kamp_relative_selector = True


def install_criterion(quality_weight: float = 1.0, beta: float = 2.0) -> None:
    if getattr(DFINECriterion, "_kamp_relative_selector", False):
        raise RuntimeError("Install relative selector criterion only once per process")
    original_forward = DFINECriterion.forward

    def patched_forward(self, outputs, targets, **kwargs):
        losses = original_forward(self, outputs, targets, **kwargs)
        target_quality = _max_iou_target(outputs["pred_boxes"].detach(), targets)
        predicted = outputs["pred_quality_logits"].squeeze(-1)
        modulation = (target_quality - predicted.sigmoid()).abs().pow(beta)
        quality_loss = F.binary_cross_entropy_with_logits(
            predicted, target_quality, reduction="none"
        ) * modulation
        losses["loss_selector_quality"] = quality_weight * quality_loss.mean()
        if not torch.isfinite(losses["loss_selector_quality"]):
            raise FloatingPointError("Nonfinite selector quality loss")
        return losses

    DFINECriterion.forward = patched_forward
    DFINECriterion._kamp_relative_selector = True


def install(mode: str, quality_weight: float = 1.0, criterion: bool = True) -> None:
    install_model(mode)
    if criterion:
        install_criterion(quality_weight)
