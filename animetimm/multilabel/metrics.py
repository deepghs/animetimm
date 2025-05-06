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
