from typing import Optional, Tuple

import numpy as np
import torch


def mcc(tp, fp, tn, fn, mean: bool = True):
    N = (tp + fn + fp + tn)
    S = (tp + fn) / N
    P = (tp + fp) / N

    numerator = (tp / N) - (S * P)
    denominator = S * P * (1 - S) * (1 - P)
    denominator = torch.clamp(denominator, min=1e-12)
    denominator = torch.sqrt(denominator)

    v = numerator / denominator
    if mean:
        v = torch.mean(v)
    return v


def f1score(tp, fp, tn, fn, alpha: float = 1.0, mean: bool = True):
    _ = tn
    numerator = (1 + alpha) * tp
    denominator = (1 + alpha) * tp + alpha * fn + fp

    mask = denominator == 0
    if mask.any():
        numerator = numerator.clone()
        denominator = denominator.clone()
        numerator[mask] = 1
        denominator[mask] = 1

    v = numerator / denominator
    if mean:
        v = torch.mean(v)
    return v


def precision(tp, fp, tn, fn, mean: bool = True):
    _ = tn
    _ = fn

    numerator = tp
    denominator = tp + fp

    mask = denominator == 0
    if mask.any():
        numerator = numerator.clone()
        denominator = denominator.clone()
        numerator[mask] = 0
        denominator[mask] = 1

    v = numerator / denominator
    if mean:
        v = torch.mean(v)
    return v


def recall(tp, fp, tn, fn, mean: bool = True):
    _ = fp
    _ = tn

    numerator = tp
    denominator = tp + fn

    mask = denominator == 0
    if mask.any():
        numerator = numerator.clone()
        denominator = denominator.clone()
        numerator[mask] = 0
        denominator[mask] = 1

    v = numerator / denominator
    if mean:
        v = torch.mean(v)
    return v


Histograms = Tuple[np.ndarray, np.ndarray]


def threshold_grid(num_thresholds: int = 100) -> np.ndarray:
    """Candidate thresholds, ``num_thresholds`` points evenly spread over ``(0, 1]``."""
    if num_thresholds < 1:
        raise ValueError(f'num_thresholds must be at least 1, got {num_thresholds!r}.')
    return np.linspace(1.0 / num_thresholds, 1, num_thresholds)


def _binarize(labels):
    """The label convention used throughout testing: truncate towards zero, then test
    for non-zero, i.e. only an exact 1.0 counts as positive for a score in [0, 1]."""
    return labels.to(torch.int32).to(torch.bool)


def _bound_dtype(sample_dtype: torch.dtype) -> torch.dtype:
    """Precision to compare scores against thresholds in: always float64.

    The thresholds come from ``np.linspace``, so they are float64 values, and the original
    scan compared with ``sample >= th`` in numpy, which promotes the float32 scores to
    float64 before comparing. Bucketizing in float32 instead would round the *thresholds*
    and flip scores that sit within one float32 ulp of one -- float32(0.03) is
    0.0299999993, which the original counts as below the 0.03 threshold but a float32
    comparison counts as at it. Real sigmoid outputs essentially never land there (zero
    disagreements over 2.5e8 measured scores), but the promotion costs a temporary in the
    batch being bucketized and removes the discrepancy entirely.
    """
    _ = sample_dtype
    return torch.float64


def _bin_indices(sample, bounds):
    """Bin index of each score: ``#{k : bounds[k] <= score}``.

    A score therefore counts as predicted-positive for exactly ``bounds[0..bin-1]``,
    which makes every threshold's confusion matrix a suffix sum over the bins.
    """
    if sample.dtype != bounds.dtype:
        sample = sample.to(bounds.dtype)
    return torch.bucketize(sample, bounds, right=True)


class StreamingThresholdHistogram:
    """Per-tag score histogram over threshold bins, accumulated one batch at a time.

    Folding each inference batch straight into a ``(num_thresholds + 1, tag_num)``
    counter means the ``(sample_num, tag_num)`` score and label matrices never have to
    be kept around, and distributed runs only have to reduce the counter rather than
    all-gather the raw tensors.

    Works identically on CPU and CUDA; the accumulator follows the device of the first
    batch handed to :meth:`update` unless ``device`` is given explicitly.
    """

    def __init__(self, tag_num: int, num_thresholds: int = 100, device=None,
                 score_dtype: Optional[torch.dtype] = None):
        self.tag_num = tag_num
        self.num_thresholds = num_thresholds
        # None means follow the first batch, the same rule the offline path uses
        self.score_dtype = score_dtype
        self._device = None
        self._bounds = None
        self._ones = None
        self.hist_all = None
        self.hist_pos = None
        if device is not None:
            self._allocate(torch.device(device))

    def _allocate(self, device):
        shape = (self.num_thresholds + 1, self.tag_num)
        self.hist_all = torch.zeros(shape, dtype=torch.int64, device=device)
        self.hist_pos = torch.zeros(shape, dtype=torch.int64, device=device)
        # normalised form of the request, e.g. 'cuda' resolves to 'cuda:0', so the
        # per-batch device check below never fires spuriously
        self._device = self.hist_all.device
        self._bounds = None
        self._ones = None

    def _bounds_for(self, sample):
        """Bin edges, built from the same np.linspace the offline path uses so both
        routes bucketize against bit-identical values."""
        dtype = self.score_dtype if self.score_dtype is not None else _bound_dtype(sample.dtype)
        if self._bounds is None or self._bounds.dtype != dtype:
            self._bounds = torch.as_tensor(threshold_grid(self.num_thresholds),
                                           dtype=dtype, device=self._device)
        return self._bounds

    def _ones_like_rows(self, rows: int):
        if self._ones is None or self._ones.shape[0] < rows:
            self._ones = torch.ones((rows, self.tag_num), dtype=torch.int64, device=self._device)
        return self._ones[:rows]

    @torch.no_grad()
    def update(self, sample, labels):
        """Fold one batch in. ``sample`` holds sigmoid scores, ``labels`` the raw labels,
        both shaped ``(batch_size, tag_num)``."""
        if sample.ndim != 2:
            raise ValueError(f'Expected a (batch_size, tag_num) batch, got {sample.ndim} dimensions.')
        if sample.shape != labels.shape:
            raise ValueError(f'Sample shape {tuple(sample.shape)} does not match '
                             f'label shape {tuple(labels.shape)}.')
        if sample.shape[-1] != self.tag_num:
            raise ValueError(f'Expected {self.tag_num} tags, got {sample.shape[-1]}.')
        if self.hist_all is None:
            self._allocate(sample.device)
        # unconditional rather than guarded: .to() of a tensor already on the target
        # device returns it unchanged, and a guard here would be a branch that a
        # CPU-only test run could never reach
        sample, labels = sample.to(self._device), labels.to(self._device)

        bins = _bin_indices(sample, self._bounds_for(sample))
        self.hist_all.scatter_add_(0, bins, self._ones_like_rows(bins.shape[0]))
        self.hist_pos.scatter_add_(0, bins, _binarize(labels).to(torch.int64))

    @torch.no_grad()
    def finalize(self, accelerator=None) -> Histograms:
        """Reduce across processes when running distributed, and hand back the
        ``(tag_num, num_thresholds + 1)`` histograms the searches consume."""
        if self.hist_all is None:
            self._allocate(self._device if self._device is not None else torch.device('cpu'))

        hist_all, hist_pos = self.hist_all, self.hist_pos
        if accelerator is not None and accelerator.num_processes > 1:
            # two (num_thresholds + 1, tag_num) int64 counters -- megabytes, not gigabytes
            hist_all = accelerator.reduce(hist_all, reduction='sum')
            hist_pos = accelerator.reduce(hist_pos, reduction='sum')
        return (hist_all.t().to(torch.float64).cpu().numpy(),
                hist_pos.t().to(torch.float64).cpu().numpy())


def build_threshold_histograms(all_sample, all_labels, num_thresholds: int = 100,
                               chunk: int = 1024, device=None) -> Histograms:
    """Offline counterpart of :class:`StreamingThresholdHistogram`, for callers that
    already hold the full ``(sample_num, tag_num)`` matrices.

    Tags are processed in chunks so peak memory stays bounded regardless of tag count.
    """
    if all_sample.ndim != 2:
        raise ValueError(f'Expected a (sample_num, tag_num) matrix, got {all_sample.ndim} dimensions.')
    if all_sample.shape != all_labels.shape:
        raise ValueError(f'Sample shape {tuple(all_sample.shape)} does not match '
                         f'label shape {tuple(all_labels.shape)}.')
    device = torch.device(device) if device is not None else all_sample.device
    tag_num = all_sample.shape[-1]
    bounds = torch.as_tensor(threshold_grid(num_thresholds),
                             dtype=_bound_dtype(all_sample.dtype), device=device)

    hist_all = np.empty((tag_num, num_thresholds + 1), dtype=np.float64)
    hist_pos = np.empty((tag_num, num_thresholds + 1), dtype=np.float64)
    width_stride = num_thresholds + 1

    for start in range(0, tag_num, chunk):
        stop = min(start + chunk, tag_num)
        width = stop - start
        sample = all_sample[:, start:stop].to(device).contiguous()
        labels = _binarize(all_labels[:, start:stop].to(device).contiguous())

        bins = _bin_indices(sample, bounds)
        # offset each column into its own slice of one flat histogram, so a single
        # bincount covers the whole chunk instead of one call per tag
        bins = bins + torch.arange(width, device=bins.device).mul(width_stride).unsqueeze(0)
        bins = bins.reshape(-1)

        size = width * width_stride
        counts_all = torch.bincount(bins, minlength=size)
        # counting only the positives' bins keeps this an integer histogram; float
        # weights would start losing counts past 2**24 samples
        counts_pos = torch.bincount(bins[labels.reshape(-1)], minlength=size)
        hist_all[start:stop] = counts_all.view(width, width_stride).cpu().numpy()
        hist_pos[start:stop] = counts_pos.view(width, width_stride).cpu().numpy()

    return hist_all, hist_pos


def _curves(hist_all, hist_pos, alpha: float):
    """Turn bin histograms into f1/precision/recall at every threshold.

    ``suffix[..., j]`` counts the scores landing in bin ``j`` or above, which is exactly
    the number of positives predicted at ``thresholds[j - 1]``.
    """
    suffix_all = np.cumsum(hist_all[..., ::-1], axis=-1)[..., ::-1]
    suffix_pos = np.cumsum(hist_pos[..., ::-1], axis=-1)[..., ::-1]
    predicted_positive, tp = suffix_all[..., 1:], suffix_pos[..., 1:]
    fp = predicted_positive - tp
    fn = hist_pos.sum(-1)[..., None] - tp

    p = tp / (tp + fp + 1e-12)
    r = tp / (tp + fn + 1e-12)
    beta_sq = alpha ** 2
    f1 = (1 + beta_sq) * p * r / (beta_sq * p + r + 1e-12)
    return f1, p, r


def _pick(f1s, pres, recs, ths):
    """Best threshold per row of a ``(n, num_thresholds)`` block.

    Ties are resolved the way the original scan did: walk forward from the argmax over
    entries whose f1/precision/recall are all numerically indistinguishable, then take
    the midpoint of the run. Expressed as a masked argmax rather than a Python loop --
    ``np.isclose`` on scalars costs tens of microseconds and dominates otherwise.
    """
    row_num, th_num = f1s.shape
    rows = np.arange(row_num)
    ma = np.argmax(f1s, axis=1)

    close = (np.isclose(f1s, f1s[rows, ma][:, None])
             & np.isclose(pres, pres[rows, ma][:, None])
             & np.isclose(recs, recs[rows, ma][:, None]))
    # first index after ma that breaks the run, or th_num when the run reaches the end
    breaks = (np.arange(th_num)[None, :] > ma[:, None]) & ~close
    mb = np.where(breaks.any(axis=1), breaks.argmax(axis=1), th_num) - 1

    return (ths[ma] + ths[mb]) / 2, f1s[rows, ma], pres[rows, ma], recs[rows, ma]


def _resolve_histograms(all_sample, all_labels, num_thresholds, histograms, device):
    if histograms is None:
        if all_sample is None or all_labels is None:
            raise ValueError('Either histograms, or both all_sample and all_labels, must be given.')
        return build_threshold_histograms(all_sample, all_labels, num_thresholds=num_thresholds,
                                          device=device)

    hist_all, hist_pos = histograms
    # a grid of the wrong size would still index without raising, and would silently
    # report thresholds taken from the wrong grid
    if hist_all.shape[-1] != num_thresholds + 1:
        raise ValueError(f'Histograms have {hist_all.shape[-1]} bins, which corresponds to '
                         f'{hist_all.shape[-1] - 1} thresholds, but num_thresholds is {num_thresholds}.')
    if hist_all.shape != hist_pos.shape:
        raise ValueError(f'Histogram shapes disagree: {hist_all.shape} versus {hist_pos.shape}.')
    return histograms


def compute_optimal_thresholds(all_sample=None, all_labels=None, alpha: float = 1.0, num_thresholds: int = 100,
                               max_workers: Optional[int] = None, histograms: Optional[Histograms] = None,
                               device=None):
    """Per-tag optimal thresholds and the f1/precision/recall attained there.

    Pass ``histograms`` from :class:`StreamingThresholdHistogram` or
    :func:`build_threshold_histograms` to reuse one pass over the data for both this
    and :func:`compute_optimal_thresholds_by_categories`.

    ``max_workers`` is accepted for backwards compatibility and ignored -- the search no
    longer uses a thread pool.
    """
    _ = max_workers
    hist_all, hist_pos = _resolve_histograms(all_sample, all_labels, num_thresholds, histograms, device)
    thresholds = threshold_grid(num_thresholds)
    f1s, pres, recs = _curves(hist_all, hist_pos, alpha)
    best_thresholds, best_f1, best_precision, best_recall = _pick(f1s, pres, recs, thresholds)
    return best_thresholds, best_f1, best_precision, best_recall


def compute_optimal_thresholds_by_categories(all_sample=None, all_labels=None, df_tags=None, alpha: float = 1.0,
                                             num_thresholds: int = 100, max_workers: Optional[int] = None,
                                             histograms: Optional[Histograms] = None, device=None):
    """Optimal threshold per tag category, micro-averaged over the tags in it.

    A category's confusion matrix at a threshold is just the sum of its tags' bin
    histograms, so this is essentially free once the per-tag histograms exist.

    ``max_workers`` is accepted for backwards compatibility and ignored.
    """
    _ = max_workers
    if df_tags is None:
        raise ValueError('df_tags is required to group tags into categories.')
    hist_all, hist_pos = _resolve_histograms(all_sample, all_labels, num_thresholds, histograms, device)
    thresholds = threshold_grid(num_thresholds)

    categories = np.asarray(df_tags['category'])
    if categories.shape[0] != hist_all.shape[0]:
        raise ValueError(f'Tag table has {categories.shape[0]} rows but the histograms '
                         f'cover {hist_all.shape[0]} tags.')

    best_f1, best_precision, best_recall, best_thresholds = {}, {}, {}, {}
    for category in sorted(set(categories.tolist())):
        mask = categories == category
        f1s, pres, recs = _curves(hist_all[mask].sum(0)[None], hist_pos[mask].sum(0)[None], alpha)
        th, f1, p, r = _pick(f1s, pres, recs, thresholds)
        best_thresholds[category] = float(th[0])
        best_f1[category] = float(f1[0])
        best_precision[category] = float(p[0])
        best_recall[category] = float(r[0])

    return best_thresholds, best_f1, best_precision, best_recall
