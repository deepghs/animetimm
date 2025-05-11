import numpy as np
import torch
from tqdm import tqdm


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


def compute_optimal_thresholds(all_sample, all_labels, alpha=1.0, num_thresholds=100):
    all_sample = all_sample.cpu().numpy()
    all_labels = all_labels.to(torch.int32).to(torch.bool).cpu().numpy()

    # Generate candidate thresholds (0 to 1)
    thresholds = np.linspace(1.0 / num_thresholds, 1, num_thresholds)

    best_f1, best_precision, best_recall, best_thresholds = [], [], [], []

    for idx in tqdm(range(all_sample.shape[-1]), desc='Scan Tags'):
        sample, labels = all_sample[..., idx], all_labels[..., idx]

        f1s, pres, recs, ths = [], [], [], []

        for th in thresholds:
            ppos = sample >= th
            tp = ((ppos == 1) & (labels == 1)).sum()
            fp = ((ppos == 1) & (labels == 0)).sum()
            fn = ((ppos == 0) & (labels == 1)).sum()

            p = tp / (tp + fp + 1e-12)
            r = tp / (tp + fn + 1e-12)
            beta_sq = alpha ** 2
            f1_numerator = (1 + beta_sq) * p * r
            f1_denominator = beta_sq * p + r + 1e-12
            f1 = f1_numerator / f1_denominator
            f1s.append(f1)
            pres.append(p)
            recs.append(r)
            ths.append(th)

        f1s = np.array(f1s)
        pres = np.array(pres)
        recs = np.array(recs)
        ths = np.array(ths)

        ma = np.argmax(f1s)
        best_f1.append(f1s[ma])
        best_precision.append(pres[ma])
        best_recall.append(recs[ma])
        best_thresholds.append(ths[ma])

    best_f1 = np.array(best_f1)
    best_precision = np.array(best_precision)
    best_recall = np.array(best_recall)
    best_thresholds = np.array(best_thresholds)

    return best_thresholds, best_f1, best_precision, best_recall
