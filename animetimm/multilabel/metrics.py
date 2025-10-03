import logging

import numpy as np
import torch

from ..utils import parallel_call


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


def compute_optimal_thresholds(all_sample, all_labels, alpha: float = 1.0, num_thresholds: int = 100,
                               max_workers: int = 32):
    all_sample = all_sample.detach().cpu().numpy()
    # print(all_labels)
    all_labels = all_labels.to(torch.int32).to(torch.bool).detach().cpu().numpy()
    # print(all_labels.to(torch.int))

    # Generate candidate thresholds (0 to 1)
    thresholds = np.linspace(1.0 / num_thresholds, 1, num_thresholds)

    best_f1, best_precision, best_recall, best_thresholds = {}, {}, {}, {}

    def _fn_cal(idx):
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

        ma = int(np.argmax(f1s).item())
        mb = int(ma) + 1
        while mb < f1s.shape[0] and np.isclose(f1s[ma], f1s[mb]) and np.isclose(pres[ma], pres[mb]) \
                and np.isclose(recs[ma], recs[mb]):
            mb += 1
        mb = mb - 1
        best_f1[idx] = f1s[ma]
        best_precision[idx] = pres[ma]
        best_recall[idx] = recs[ma]
        best_thresholds[idx] = (ths[ma] + ths[mb]) / 2

    parallel_call(
        iterable=range(all_sample.shape[-1]),
        fn=_fn_cal,
        desc='Scan Tags',
        max_workers=max_workers,
    )

    best_f1 = np.array([best_f1[idx] for idx in range(all_sample.shape[-1])])
    best_precision = np.array([best_precision[idx] for idx in range(all_sample.shape[-1])])
    best_recall = np.array([best_recall[idx] for idx in range(all_sample.shape[-1])])
    best_thresholds = np.array([best_thresholds[idx] for idx in range(all_sample.shape[-1])])

    return best_thresholds, best_f1, best_precision, best_recall


def compute_optimal_thresholds_by_categories(all_sample, all_labels, df_tags,
                                             alpha: float = 1.0, num_thresholds: int = 100, max_workers: int = 16):
    all_sample = all_sample.detach().cpu().numpy()
    # print(all_labels)
    all_labels = all_labels.to(torch.int32).to(torch.bool).detach().cpu().numpy()
    # print(all_labels.to(torch.int))

    # Generate candidate thresholds (0 to 1)
    thresholds = np.linspace(1.0 / num_thresholds, 1, num_thresholds)

    best_f1, best_precision, best_recall, best_thresholds = {}, {}, {}, {}

    for category in sorted(set(df_tags['category'])):
        logging.info(f'Scanning for category {category!r} ...')
        mask = df_tags['category'] == category
        sample, labels = all_sample[..., mask], all_labels[..., mask]

        f1s, pres, recs, ths = {}, {}, {}, {}

        def _fn_call(idx):
            th = thresholds[idx]
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
            f1s[idx] = f1
            pres[idx] = p
            recs[idx] = r
            ths[idx] = th

        parallel_call(
            iterable=range(thresholds.shape[0]),
            fn=_fn_call,
            desc=f'Scan Thresholds For #{category}',
            max_workers=max_workers,
        )

        f1s = np.array([f1s[idx] for idx in range(thresholds.shape[0])])
        pres = np.array([pres[idx] for idx in range(thresholds.shape[0])])
        recs = np.array([recs[idx] for idx in range(thresholds.shape[0])])
        ths = np.array([ths[idx] for idx in range(thresholds.shape[0])])

        ma = int(np.argmax(f1s).item())
        mb = int(ma) + 1
        while mb < f1s.shape[0] and np.isclose(f1s[ma], f1s[mb]) and np.isclose(pres[ma], pres[mb]) \
                and np.isclose(recs[ma], recs[mb]):
            mb += 1
        mb = mb - 1
        best_f1[category] = float(f1s[ma])
        best_precision[category] = float(pres[ma])
        best_recall[category] = float(recs[ma])
        best_thresholds[category] = float((ths[ma] + ths[mb]) / 2)

    return best_thresholds, best_f1, best_precision, best_recall
