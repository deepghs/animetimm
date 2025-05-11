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


def compute_optimal_thresholds(all_sample, all_labels, alpha=1.0, num_thresholds=100, batch_size=100000):
    """
    Calculate optimal thresholds for multi-label classification maximizing F-beta score

    Args:
    - all_sample: Model output probabilities, shape [num_samples, num_tags], torch.float32
    - all_labels: Ground truth labels, shape [num_samples, num_tags], torch.float32 (0.0 or 1.0)
    - alpha: Beta parameter for F-beta score calculation
    - num_thresholds: Number of candidate thresholds (default: 100)
    - batch_size: Number of samples per processing batch (adjust based on memory)

    Returns:
    - best_thresholds: Optimal thresholds per tag, shape [num_tags]
    - best_f1: Best F-beta scores, shape [num_tags]
    - best_precision: Precision values at optimal thresholds, shape [num_tags]
    - best_recall: Recall values at optimal thresholds, shape [num_tags]
    """
    device = all_sample.device
    num_samples, num_tags = all_sample.shape

    # Precompute total positive samples per tag
    sum_labels = all_labels.sum(dim=0)  # [num_tags]

    # Generate candidate thresholds (0 to 1)
    thresholds = torch.linspace(0, 1, steps=num_thresholds, device=device)

    # Initialize storage for best metrics
    best_f1 = torch.zeros(num_tags, device=device)
    best_precision = torch.zeros_like(best_f1)
    best_recall = torch.zeros_like(best_f1)
    best_thresholds = torch.zeros_like(best_f1)

    # Process each candidate threshold
    for t in thresholds:
        # Initialize counters for current threshold
        tp_total = torch.zeros(num_tags, device=device)
        fp_total = torch.zeros(num_tags, device=device)

        # Process samples in batches
        for i in range(0, num_samples, batch_size):
            batch_slice = slice(i, min(i + batch_size, num_samples))
            sample_batch = all_sample[batch_slice]  # [batch_size, num_tags]
            label_batch = all_labels[batch_slice]  # [batch_size, num_tags]

            # Calculate TP and FP for current batch
            pred_mask = sample_batch >= t
            tp_batch = (pred_mask & label_batch.bool()).sum(dim=0).float()
            fp_batch = (pred_mask & ~label_batch.bool()).sum(dim=0).float()

            # Accumulate batch results
            tp_total += tp_batch
            fp_total += fp_batch

        # Calculate derived metrics
        fn_total = sum_labels - tp_total
        precision = tp_total / (tp_total + fp_total + 1e-12)  # Add epsilon to avoid division by zero
        recall = tp_total / (tp_total + fn_total + 1e-12)

        # Calculate F-beta score (beta = alpha)
        beta_sq = alpha ** 2
        f1_numerator = (1 + beta_sq) * precision * recall
        f1_denominator = beta_sq * precision + recall + 1e-12
        f1 = f1_numerator / f1_denominator

        # Update best metrics where F1 improves
        better_mask = f1 > best_f1
        best_f1[better_mask] = f1[better_mask]
        best_precision[better_mask] = precision[better_mask]
        best_recall[better_mask] = recall[better_mask]
        best_thresholds[better_mask] = t

    return best_thresholds, best_f1, best_precision, best_recall
