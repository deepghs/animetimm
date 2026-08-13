"""Loss functions for multi-label tagging under one-sided label noise.

Danbooru-style tag data is high-precision and low-recall: a tag that is present
is almost always correct, a tag that is absent is often just missing.  Training
with plain BCE therefore feeds the model a large, *systematic* stream of false
negatives, and the rate differs enormously between tags -- an audit of dbv4 put
the omission rate of `1girl`-class tags near zero and that of composition tags
around three quarters.

Every loss here returns a per-element tensor of shape ``(B, C)`` with no
reduction, so the caller keeps its existing per-tag weighting and reduction.
Every loss also accepts *soft* targets, because mixup is on by default in this
codebase and ``labels`` arrive fractional.

    bce     exact reproduction of ``BCEWithLogitsLoss(reduction='none')``;
            the default, so nothing changes unless it is asked to
    asl     Asymmetric Loss (Ridnik et al., ICCV'21).  One extra focusing
            parameter for negatives plus probability shifting, which is the
            de-facto baseline for booru taggers
    pasl    Partial-ASL with class-aware selective treatment, adapted from
            Ben-Baruch et al., CVPR'22 (arXiv:2110.10955)

Adapting P-ASL to this dataset needs one deliberate change.  In the original,
labels are explicitly annotated / un-annotated, and the class-prior gate can
drop a whole class's un-annotated entries because most entries of that class are
annotated anyway.  Here *every* non-positive is an assumed negative, so dropping
a tag's negatives wholesale would mean never learning when the tag is absent.
The per-tag signal is therefore expressed as a continuous interpolation of the
negative focusing parameter between ``gamma_neg`` and ``gamma_unann``, and hard
dropping is reserved for the per-sample top-k gate, which only removes the
specific cells the model itself finds suspicious.
"""
from typing import Optional

import torch
from torch import nn

__all__ = [
    'BCELoss', 'AsymmetricLoss', 'PartialAsymmetricLoss', 'build_loss',
    'LOSS_NAMES',
]

LOSS_NAMES = ('bce', 'asl', 'pasl')
_EPS = 1e-8


class BCELoss(nn.Module):
    """``BCEWithLogitsLoss(reduction='none')`` behind the common interface."""

    def __init__(self):
        super().__init__()
        self._inner = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._inner(logits, targets.to(logits.dtype))

    def extra_repr(self) -> str:
        return 'mode=bce'


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss, following the reference implementation.

    ``gamma_neg > gamma_pos`` decays the contribution of easy negatives faster
    than that of positives, which matters when positives are rare.  ``clip``
    shifts the negative probability and hard-zeroes negatives the model already
    scores below the margin -- the part that actually protects against unlabeled
    positives, since a confidently-predicted "negative" is exactly what a missing
    label looks like.

    The focusing weight is detached by default, matching
    ``AsymmetricLossOptimized``; the gradient then flows only through the log
    terms.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                 clip: float = 0.05, detach_focus: bool = True, eps: float = _EPS):
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.detach_focus = bool(detach_focus)
        self.eps = float(eps)

    def _terms(self, logits, targets):
        # log-probabilities come from logsigmoid rather than log(sigmoid(.)):
        # the latter loses the low tail to float32 rounding well before the
        # clamp would catch it, which is exactly the regime a confident negative
        # sits in.  With clip=0 this also makes the loss bit-comparable to
        # BCEWithLogitsLoss, so `asl` degenerates to the old path exactly.
        y = targets.to(logits.dtype)
        p_pos = torch.sigmoid(logits)
        log_pos = nn.functional.logsigmoid(logits)
        if self.clip > 0:
            p_neg = (1.0 - p_pos + self.clip).clamp(max=1.0)
            log_neg = torch.log(p_neg.clamp(min=self.eps))
        else:
            p_neg = 1.0 - p_pos
            log_neg = nn.functional.logsigmoid(-logits)
        return y, p_pos, p_neg, log_pos, log_neg

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        y, p_pos, p_neg, log_pos, log_neg = self._terms(logits, targets)
        with torch.set_grad_enabled(not self.detach_focus):
            w_pos = (1.0 - p_pos).clamp(min=0).pow(self.gamma_pos) \
                if self.gamma_pos else torch.ones_like(p_pos)
            w_neg = (1.0 - p_neg).clamp(min=0).pow(self.gamma_neg) \
                if self.gamma_neg else torch.ones_like(p_neg)
        if self.detach_focus:
            w_pos, w_neg = w_pos.detach(), w_neg.detach()
        return -(y * w_pos * log_pos + (1.0 - y) * w_neg * log_neg)

    def extra_repr(self) -> str:
        return (f'mode=asl, gamma_neg={self.gamma_neg}, gamma_pos={self.gamma_pos}, '
                f'clip={self.clip}')


class PartialAsymmetricLoss(AsymmetricLoss):
    """P-ASL with class-aware and sample-aware selective treatment of negatives.

    Three mechanisms, each independently switchable so their contributions can be
    told apart:

    per-tag focusing (``neg_gamma``)
        A length-``C`` vector giving each tag its own negative focusing
        parameter, expected to run between ``gamma_neg`` for tags whose absence
        is trustworthy and ``gamma_unann`` for tags whose absence is mostly
        missing data.  Build it with :func:`neg_gamma_from_scores`.

    per-sample top-k gate (``ignore_topk``)
        Within each sample, the ``k`` highest-scoring *negative* entries are
        dropped from the loss.  This is the mechanism that survives a noisy
        per-tag estimate: it acts on the cells the model finds suspicious rather
        than on a whole tag.  Positives are never gated.

    diligence modulation (``diligence_*``)
        Danbooru posts differ hugely in how thoroughly they were tagged, and the
        thoroughness of a training sample is observable for free -- it is the
        number of positive labels the sample carries.  A sample with few labels
        has unreliable negatives, so ``k`` is scaled up for it.  This lever has
        no analogue in the original paper; it exists because this dataset
        happens to expose the annotator's effort.
    """

    def __init__(self, gamma_neg: float = 2.0, gamma_pos: float = 0.0,
                 gamma_unann: float = 7.0, clip: float = 0.05,
                 neg_gamma: Optional[torch.Tensor] = None,
                 ignore_topk: int = 0,
                 diligence_modulate: bool = False,
                 diligence_lo: float = 12.0, diligence_hi: float = 48.0,
                 diligence_max_scale: float = 2.0,
                 detach_focus: bool = True, eps: float = _EPS):
        super().__init__(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip,
                         detach_focus=detach_focus, eps=eps)
        self.gamma_unann = float(gamma_unann)
        self.ignore_topk = int(ignore_topk)
        self.diligence_modulate = bool(diligence_modulate)
        self.diligence_lo = float(diligence_lo)
        self.diligence_hi = float(diligence_hi)
        self.diligence_max_scale = float(diligence_max_scale)
        if neg_gamma is not None:
            self.register_buffer('neg_gamma', neg_gamma.float(), persistent=False)
        else:
            self.neg_gamma = None

    def _per_sample_k(self, targets: torch.Tensor) -> torch.Tensor:
        """k for each row, scaled up where the sample looks sparsely tagged."""
        k = torch.full((targets.shape[0],), float(self.ignore_topk),
                       device=targets.device)
        if not self.diligence_modulate:
            return k
        # thoroughness of this sample = how many labels its annotator supplied
        n_pos = targets.gt(0.5).sum(dim=1).to(k.dtype)
        span = max(self.diligence_hi - self.diligence_lo, 1e-6)
        frac = ((n_pos - self.diligence_lo) / span).clamp(0.0, 1.0)
        # sparsely tagged (frac -> 0) gets the full scale, thoroughly tagged
        # (frac -> 1) gets no extra gating
        return k * (1.0 + (self.diligence_max_scale - 1.0) * (1.0 - frac))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        y, p_pos, p_neg, log_pos, log_neg = self._terms(logits, targets)

        gamma_neg = self.neg_gamma.to(logits.dtype).unsqueeze(0) \
            if self.neg_gamma is not None else \
            torch.full_like(p_neg, self.gamma_neg)

        with torch.set_grad_enabled(not self.detach_focus):
            w_pos = (1.0 - p_pos).clamp(min=0).pow(self.gamma_pos) \
                if self.gamma_pos else torch.ones_like(p_pos)
            w_neg = (1.0 - p_neg).clamp(min=0).pow(gamma_neg)
        if self.detach_focus:
            w_pos, w_neg = w_pos.detach(), w_neg.detach()

        keep = torch.ones_like(p_pos)
        if self.ignore_topk > 0:
            with torch.no_grad():
                is_neg = (y <= 0.5)
                # rank negatives by predicted probability, descending
                order = torch.argsort(
                    torch.where(is_neg, p_pos, torch.full_like(p_pos, -1.0)),
                    dim=1, descending=True)
                rank = torch.empty_like(order)
                rank.scatter_(1, order,
                              torch.arange(p_pos.shape[1], device=p_pos.device)
                              .expand_as(order))
                k = self._per_sample_k(y).unsqueeze(1)
                keep = torch.where(is_neg & (rank < k),
                                   torch.zeros_like(p_pos), keep)

        return -(y * w_pos * log_pos + (1.0 - y) * keep * w_neg * log_neg)

    def extra_repr(self) -> str:
        return (f'mode=pasl, gamma_neg={self.gamma_neg}, gamma_pos={self.gamma_pos}, '
                f'gamma_unann={self.gamma_unann}, clip={self.clip}, '
                f'per_tag_gamma={self.neg_gamma is not None}, '
                f'ignore_topk={self.ignore_topk}, '
                f'diligence_modulate={self.diligence_modulate}')


def neg_gamma_from_scores(scores, gamma_neg: float, gamma_unann: float,
                          lo_quantile: float = 0.1,
                          hi_quantile: float = 0.9) -> torch.Tensor:
    """Map a per-tag label-reliability score onto a per-tag negative focusing.

    ``scores`` is expected to rise with how *trustworthy* a tag's absence is --
    an estimate of the tag's label recall works, since a tag that is nearly
    always applied when it is true has meaningful negatives.  The mapping is
    rank-free but robust: it is linear between the given quantiles and clamped
    outside them, so a handful of extreme tags cannot stretch the scale.

    Returns ``gamma_unann`` for the least reliable tags and ``gamma_neg`` for the
    most reliable ones.
    """
    s = torch.as_tensor(scores, dtype=torch.float32).flatten()
    finite = s[torch.isfinite(s)]
    if finite.numel() == 0:
        return torch.full_like(s, gamma_neg)
    lo = torch.quantile(finite, lo_quantile)
    hi = torch.quantile(finite, hi_quantile)
    frac = ((s - lo) / (hi - lo).clamp(min=1e-6)).clamp(0.0, 1.0)
    frac = torch.where(torch.isfinite(frac), frac, torch.ones_like(frac))
    return gamma_unann + (gamma_neg - gamma_unann) * frac


def build_loss(name: str = 'bce', **kwargs) -> nn.Module:
    """Factory used by the training entry point.  ``bce`` keeps the old path."""
    name = (name or 'bce').lower()
    if name == 'bce':
        return BCELoss()
    if name == 'asl':
        return AsymmetricLoss(**kwargs)
    if name == 'pasl':
        return PartialAsymmetricLoss(**kwargs)
    raise ValueError(f'Unknown loss {name!r}, expected one of {LOSS_NAMES!r}.')
