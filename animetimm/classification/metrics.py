import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support


def calculate_topk_accuracy(y_true, y_pred_probs, k=5):
    batch_size = len(y_true)
    topk_indices = np.argsort(y_pred_probs, axis=1)[:, -k:]
    correct = 0
    for i in range(batch_size):
        if y_true[i] in topk_indices[i]:
            correct += 1
    return correct / batch_size


def calculate_macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def calculate_micro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')


def calculate_label_level_metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(0, num_classes),
        average=None,
        zero_division=0.0
    )
    return f1, precision, recall


if __name__ == '__main__':
    num_classes = 10
    batch_size = 5
    logits = torch.randn(batch_size, num_classes)
    probs = torch.softmax(logits, dim=-1)
    labels = torch.argmax(probs, dim=-1)
    labels_y = torch.randint(0, num_classes, (batch_size,))
    mask = torch.randn_like(labels, dtype=torch.float32) < -0.5
    print(mask)
    labels[mask] = labels_y[mask]
    print(torch.argsort(logits, dim=-1))

    print(probs, labels, torch.argmax(probs, dim=-1))
    print('top-1', calculate_topk_accuracy(labels, probs, k=1))
    print('top-5', calculate_topk_accuracy(labels, probs, k=5))
    print('macro_f1', calculate_macro_f1(labels, torch.argmax(probs, dim=-1)))
    print('micro_f1', calculate_micro_f1(labels, torch.argmax(probs, dim=-1)))

    # class level metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        torch.argmax(probs, dim=-1),
        labels=np.arange(0, num_classes),
        average=None,
        zero_division=0.0
    )
    print(precision)
    print(recall)
    print(f1)
    print(support)
