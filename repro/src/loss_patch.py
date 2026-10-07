# MAL adapted from DEIM criterion; Copyright (c) 2024 The DEIM Authors.
# Modified for KAMP: preserve D-FINE matching; install MAL as an optional loss patch.
# See ../licenses/DEIM_LICENSE and ../THIRD_PARTY_NOTICES.txt.
"""Minimal loss ablations; original model and original losses stay intact."""
import torch
import torch.nn.functional as F
from src.zoo.dfine.dfine_criterion import DFINECriterion
from src.zoo.dfine.box_ops import box_cxcywh_to_xyxy, box_iou


def size_loss(pred, target):
    # Normalized coordinates, robust relative size error; center is unsupervised here.
    error = torch.log((pred[:, 2:] + 1e-6) / (target[:, 2:] + 1e-6))
    return F.huber_loss(error, torch.zeros_like(error), delta=1.0, reduction='sum') / max(len(pred), 1)


def mal(self, outputs, targets, indices, num_boxes, values=None):
    # DEIM official MAL (gamma=2, mal_alpha=None); keep original D-FINE matching.
    idx = self._get_src_permutation_idx(indices)
    logits = outputs['pred_logits']
    if values is None:
        gt = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)])
        values = box_iou(box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), box_cxcywh_to_xyxy(gt))[0].diag().detach()
    labels = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)])
    classes = torch.full(logits.shape[:2], self.num_classes, dtype=torch.int64, device=logits.device)
    classes[idx] = labels
    positive = F.one_hot(classes, self.num_classes + 1)[..., :-1]
    quality = torch.zeros_like(classes, dtype=logits.dtype)
    quality[idx] = values.to(logits.dtype)
    target = (quality.unsqueeze(-1) * positive).pow(self.gamma)
    weight = logits.sigmoid().detach().pow(self.gamma) * (1-positive) + positive
    loss = F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction='none')
    return {'loss_mal': loss.mean(1).sum() * logits.shape[1] / num_boxes}


def install(size_weight=0.0, use_mal=False):
    if getattr(DFINECriterion, '_kamp_v3', False):
        raise RuntimeError('Install loss patch only once per process')
    original = DFINECriterion.forward
    if use_mal:
        DFINECriterion.loss_labels_vfl = mal
    def forward(self, outputs, targets, **kwargs):
        if use_mal:
            self.weight_dict['loss_mal'] = self.weight_dict['loss_vfl']
        for key in ('pred_boxes', 'pred_logits'):
            if not torch.isfinite(outputs[key]).all():
                raise FloatingPointError('Nonfinite model output: '+key)
        result = original(self, outputs, targets, **kwargs)
        if size_weight:
            # Final regular decoder only, final O2O matching, not GO union or DN.
            indices = self.matcher({k:v for k,v in outputs.items() if 'aux' not in k}, targets)['indices']
            idx = self._get_src_permutation_idx(indices)
            gt = torch.cat([t['boxes'][j] for t,(_,j) in zip(targets,indices)])
            result['loss_size'] = size_weight * size_loss(outputs['pred_boxes'][idx], gt)
        if not all(torch.isfinite(v).all() for v in result.values()):
            raise FloatingPointError('Nonfinite loss')
        return result
    DFINECriterion.forward = forward
    DFINECriterion._kamp_v3 = True
