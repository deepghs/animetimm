"""Behavioural lock for the threshold search.

The reference implementations below are the original brute-force scans, transcribed
verbatim from the version of ``animetimm/multilabel/metrics.py`` that preceded the
histogram rewrite. Every test asserts the fast path reproduces them *exactly*, so any
future change that shifts a published threshold has to break a test first.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from animetimm.multilabel.metrics import (
    StreamingThresholdHistogram,
    build_threshold_histograms,
    compute_optimal_thresholds,
    compute_optimal_thresholds_by_categories,
    f1score,
    mcc,
    precision,
    recall,
    threshold_grid,
)


# ---------------------------------------------------------------- reference (oracle)

def reference_optimal_thresholds(all_sample, all_labels, alpha: float = 1.0, num_thresholds: int = 100):
    """Original per-tag scan: for every tag, evaluate every threshold from scratch."""
    all_sample = all_sample.detach().cpu().numpy()
    all_labels = all_labels.to(torch.int32).to(torch.bool).detach().cpu().numpy()
    thresholds = np.linspace(1.0 / num_thresholds, 1, num_thresholds)

    best_f1, best_precision, best_recall, best_thresholds = {}, {}, {}, {}
    for idx in range(all_sample.shape[-1]):
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
            f1 = (1 + beta_sq) * p * r / (beta_sq * p + r + 1e-12)
            f1s.append(f1)
            pres.append(p)
            recs.append(r)
            ths.append(th)

        f1s, pres, recs, ths = np.array(f1s), np.array(pres), np.array(recs), np.array(ths)
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

    tag_num = all_sample.shape[-1]
    return (np.array([best_thresholds[i] for i in range(tag_num)]),
            np.array([best_f1[i] for i in range(tag_num)]),
            np.array([best_precision[i] for i in range(tag_num)]),
            np.array([best_recall[i] for i in range(tag_num)]))


def reference_optimal_thresholds_by_categories(all_sample, all_labels, df_tags, alpha: float = 1.0,
                                               num_thresholds: int = 100):
    """Original per-category scan: micro-average over every tag in the category."""
    all_sample = all_sample.detach().cpu().numpy()
    all_labels = all_labels.to(torch.int32).to(torch.bool).detach().cpu().numpy()
    thresholds = np.linspace(1.0 / num_thresholds, 1, num_thresholds)

    best_f1, best_precision, best_recall, best_thresholds = {}, {}, {}, {}
    for category in sorted(set(df_tags['category'])):
        mask = df_tags['category'] == category
        sample, labels = all_sample[..., mask], all_labels[..., mask]

        f1s, pres, recs, ths = [], [], [], []
        for th in thresholds:
            ppos = sample >= th
            tp = ((ppos == 1) & (labels == 1)).sum()
            fp = ((ppos == 1) & (labels == 0)).sum()
            fn = ((ppos == 0) & (labels == 1)).sum()

            p = tp / (tp + fp + 1e-12)
            r = tp / (tp + fn + 1e-12)
            beta_sq = alpha ** 2
            f1 = (1 + beta_sq) * p * r / (beta_sq * p + r + 1e-12)
            f1s.append(f1)
            pres.append(p)
            recs.append(r)
            ths.append(th)

        f1s, pres, recs, ths = np.array(f1s), np.array(pres), np.array(recs), np.array(ths)
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


# ---------------------------------------------------------------------- fixtures

def make_case(sample_num=512, tag_num=48, seed=0, positive_rate=0.06):
    """Scores that look like sigmoid outputs: mostly near zero, positives pushed up."""
    rng = np.random.default_rng(seed)
    labels = (rng.random((sample_num, tag_num)) < positive_rate).astype(np.float32)
    logit = rng.standard_normal((sample_num, tag_num)).astype(np.float32) - 2.0 + 4.0 * labels
    samples = (1.0 / (1.0 + np.exp(-logit))).astype(np.float32)
    return torch.from_numpy(samples), torch.from_numpy(labels)


def make_degenerate_case():
    """Corner cases that make the tie-walk and the zero-denominator guards matter."""
    sample_num, tag_num = 256, 8
    rng = np.random.default_rng(7)
    samples = rng.random((sample_num, tag_num)).astype(np.float32)
    labels = (rng.random((sample_num, tag_num)) < 0.3).astype(np.float32)

    labels[:, 0] = 0.0                      # tag never positive
    labels[:, 1] = 1.0                      # tag always positive
    samples[:, 2] = 0.0                     # scores pinned to the bottom
    samples[:, 3] = 1.0                     # scores pinned to the top
    samples[:, 4] = 0.5                     # every score exactly on a bin edge
    labels[:, 5] = 1.0
    samples[:, 5] = 1.0                     # perfectly separable
    samples[:, 6] = 1.0 - labels[:, 6]      # perfectly anti-correlated
    return torch.from_numpy(samples), torch.from_numpy(labels)


def category_frame(tag_num, groups=(0, 4, 9)):
    rng = np.random.default_rng(11)
    cats = rng.choice(np.asarray(groups), size=tag_num)
    cats[:len(groups)] = np.asarray(groups)  # guarantee every category is populated
    return pd.DataFrame({'category': cats})


def assert_all_identical(expected, actual, names=('threshold', 'f1', 'precision', 'recall')):
    for name, a, b in zip(names, expected, actual):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        assert a.shape == b.shape, f'{name}: shape {a.shape} != {b.shape}'
        assert np.array_equal(a, b), (
            f'{name}: {int((a != b).sum())}/{a.size} entries differ, '
            f'max|diff|={np.abs(a - b).max():.3e}'
        )


# ------------------------------------------------------------------------- tests

@pytest.mark.unittest
class TestHandWorkedExamples:
    """Absolute correctness, not just agreement with the previous implementation.

    Every expected value below is derived by hand in the comments, so these tests would
    catch an error that the old code and the new code happen to share. They need no GPU,
    which is the point: correctness is decidable on a CPU-only machine.
    """

    # 5 samples, 1 tag, K=10 -> thresholds 0.1, 0.2, ... 1.0
    SCORES = [0.05, 0.15, 0.35, 0.55, 0.95]
    LABELS = [0, 0, 1, 1, 1]

    def _one_tag(self):
        return (torch.tensor([[s] for s in self.SCORES], dtype=torch.float32),
                torch.tensor([[float(v)] for v in self.LABELS], dtype=torch.float32))

    def test_bin_indices_are_the_count_of_thresholds_at_or_below_the_score(self):
        # bin(s) = #{k : threshold[k] <= s}
        #   0.05 -> 0 thresholds   0.15 -> 1 (0.1)     0.35 -> 3 (0.1,0.2,0.3)
        #   0.55 -> 5 (..0.5)      0.95 -> 9 (..0.9)
        hist_all, hist_pos = build_threshold_histograms(*self._one_tag(), num_thresholds=10)
        assert {i: int(v) for i, v in enumerate(hist_all[0]) if v} == {0: 1, 1: 1, 3: 1, 5: 1, 9: 1}
        # only the last three samples are positive
        assert {i: int(v) for i, v in enumerate(hist_pos[0]) if v} == {3: 1, 5: 1, 9: 1}

    def test_optimal_threshold_matches_the_hand_derivation(self):
        # th=0.1 -> tp=3 fp=1 fn=0 -> p=3/4 r=1    f1=0.857142...
        # th=0.2 -> tp=3 fp=0 fn=0 -> p=1   r=1    f1=1.0        <- joint best
        # th=0.3 -> tp=3 fp=0 fn=0 -> p=1   r=1    f1=1.0        <- joint best
        # th=0.4 -> tp=2 fp=0 fn=1 -> p=1   r=2/3  f1=0.8
        # ... decreasing from there, and 0 at th=1.0
        # the run of identical (f1, p, r) spans 0.2 and 0.3, so the midpoint is 0.25
        th, f1, p, r = compute_optimal_thresholds(*self._one_tag(), num_thresholds=10)
        assert th[0] == pytest.approx(0.25)
        assert f1[0] == pytest.approx(1.0)
        assert p[0] == pytest.approx(1.0)
        assert r[0] == pytest.approx(1.0)

    def test_tie_run_spanning_many_thresholds_takes_the_midpoint(self):
        # one positive sample scoring 0.95: every threshold from 0.1 to 0.9 predicts it
        # correctly (f1 = 1.0), and 1.0 misses it (f1 = 0). The run is 0.1 .. 0.9, so the
        # midpoint is (0.1 + 0.9) / 2 = 0.5
        samples = torch.tensor([[0.95]], dtype=torch.float32)
        labels = torch.tensor([[1.0]], dtype=torch.float32)
        th, f1, _, _ = compute_optimal_thresholds(samples, labels, num_thresholds=10)
        assert th[0] == pytest.approx(0.5)
        assert f1[0] == pytest.approx(1.0)

    def test_category_result_is_the_micro_average_over_its_tags(self):
        # tag0: scores [0.15, 0.35, 0.55], labels [0, 1, 1]
        # tag1: scores [0.25, 0.45, 0.05], labels [1, 1, 0]
        # th=0.1 -> tp=4 fp=1 fn=0 -> p=0.8 r=1    f1=0.888...
        # th=0.2 -> tp=4 fp=0 fn=0 -> p=1   r=1    f1=1.0     <- unique best
        # th=0.3 -> tp=3 fp=0 fn=1 -> p=1   r=0.75 f1=0.857...
        samples = torch.tensor([[0.15, 0.25], [0.35, 0.45], [0.55, 0.05]], dtype=torch.float32)
        labels = torch.tensor([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        df_tags = pd.DataFrame({'category': [7, 7]})
        th, f1, p, r = compute_optimal_thresholds_by_categories(samples, labels, df_tags,
                                                                num_thresholds=10)
        assert th[7] == pytest.approx(0.2)
        assert f1[7] == pytest.approx(1.0)
        assert p[7] == pytest.approx(1.0)
        assert r[7] == pytest.approx(1.0)

    def test_perfect_and_inverted_separation(self):
        # tag 0 is perfectly separable at 0.5; tag 1 has scores exactly inverted, so no
        # threshold does better than predicting everything positive
        samples = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]], dtype=torch.float32)
        labels = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
        _, f1, p, r = compute_optimal_thresholds(samples, labels, num_thresholds=10)
        assert f1[0] == pytest.approx(1.0) and p[0] == pytest.approx(1.0) and r[0] == pytest.approx(1.0)
        # inverted: best is threshold 0.1, catching both positives plus both negatives,
        # so tp=2 fp=2 fn=0 -> p=0.5, r=1, f1=2/3
        assert f1[1] == pytest.approx(2 / 3)
        assert p[1] == pytest.approx(0.5)
        assert r[1] == pytest.approx(1.0)


@pytest.mark.unittest
class TestConfusionMetrics:
    """`mcc`, `f1score`, `precision` and `recall` feed test_metrics.json directly.

    Checked against values worked out by hand, and MCC additionally against the textbook
    formula, since the implementation uses a rescaled form that is not obviously the same.
    """

    @staticmethod
    def counts(tp, fp, tn, fn):
        return tuple(torch.tensor([float(v)]) for v in (tp, fp, tn, fn))

    def test_precision_recall_f1_by_hand(self):
        # tp=3 fp=3 tn=4 fn=1
        args = self.counts(3, 3, 4, 1)
        assert precision(*args).item() == pytest.approx(3 / 6)
        assert recall(*args).item() == pytest.approx(3 / 4)
        assert f1score(*args).item() == pytest.approx(2 * 3 / (2 * 3 + 1 + 3), rel=1e-6)

    def test_mcc_equals_the_textbook_formula(self):
        for tp, fp, tn, fn in [(3, 3, 4, 1), (10, 2, 50, 8), (1, 0, 99, 0), (25, 25, 25, 25)]:
            expected = ((tp * tn - fp * fn)
                        / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
            assert mcc(*self.counts(tp, fp, tn, fn)).item() == pytest.approx(expected, rel=1e-5), \
                f'tp={tp} fp={fp} tn={tn} fn={fn}'

    def test_perfect_and_worst_case(self):
        assert mcc(*self.counts(50, 0, 50, 0)).item() == pytest.approx(1.0, rel=1e-5)
        assert f1score(*self.counts(50, 0, 50, 0)).item() == pytest.approx(1.0, rel=1e-6)
        # every prediction inverted
        assert mcc(*self.counts(0, 50, 0, 50)).item() == pytest.approx(-1.0, rel=1e-5)

    def test_alpha_is_beta_squared_not_beta(self):
        """Locks a quirk worth knowing: `f1score`'s alpha is the *squared* beta of the
        usual F-beta, while `compute_optimal_thresholds` squares its own alpha. Both are
        called with alpha=1.0 in this repository, where the two conventions coincide."""
        args = self.counts(3, 3, 4, 1)   # p = 0.5, r = 0.75
        p_, r_ = 0.5, 0.75
        f_beta_sq_4 = (1 + 4) * p_ * r_ / (4 * p_ + r_)
        f_beta_4 = (1 + 16) * p_ * r_ / (16 * p_ + r_)
        assert f1score(*args, alpha=4).item() == pytest.approx(f_beta_sq_4, rel=1e-6)
        assert f1score(*args, alpha=4).item() != pytest.approx(f_beta_4, rel=1e-6)

    @pytest.mark.parametrize('tp,fp,tn,fn', [
        (0, 0, 10, 0),    # nothing predicted positive, nothing actually positive
        (0, 10, 0, 0),    # everything predicted positive, nothing actually positive
        (0, 0, 0, 10),    # nothing predicted positive, everything actually positive
        (10, 0, 0, 0),    # everything positive and every prediction right
    ])
    def test_degenerate_counts_do_not_produce_nan(self, tp, fp, tn, fn):
        for fn_ in (mcc, f1score, precision, recall):
            value = fn_(*self.counts(tp, fp, tn, fn)).item()
            assert not np.isnan(value), f'{fn_.__name__} returned NaN for {(tp, fp, tn, fn)}'

    def test_empty_confusion_matrix_yields_nan_mcc(self):
        """Documents pre-existing behaviour rather than endorsing it: with no samples at
        all, `mcc` divides 0 by 0 while the other three return finite values. This is
        unreachable in `test.py` — micro counts total over every sample and tag, macro
        counts total over every sample — so it is recorded, not fixed, here."""
        empty = self.counts(0, 0, 0, 0)
        assert np.isnan(mcc(*empty).item())
        for fn_ in (f1score, precision, recall):
            assert not np.isnan(fn_(*empty).item())

    def test_mean_false_returns_per_entry_values(self):
        tp = torch.tensor([3.0, 10.0])
        fp = torch.tensor([3.0, 2.0])
        tn = torch.tensor([4.0, 50.0])
        fn = torch.tensor([1.0, 8.0])
        per_entry = precision(tp, fp, tn, fn, mean=False)
        assert per_entry.shape == (2,)
        np.testing.assert_allclose(per_entry.numpy(), [3 / 6, 10 / 12], rtol=1e-6)
        assert precision(tp, fp, tn, fn).item() == pytest.approx((3 / 6 + 10 / 12) / 2, rel=1e-6)


@pytest.mark.unittest
class TestThresholdGrid:
    def test_matches_documented_grid(self):
        grid = threshold_grid(100)
        assert grid.shape == (100,)
        np.testing.assert_array_equal(grid, np.linspace(0.01, 1, 100))

    @pytest.mark.parametrize('num_thresholds', [1, 2, 10, 100, 257])
    def test_endpoints(self, num_thresholds):
        grid = threshold_grid(num_thresholds)
        assert grid.shape == (num_thresholds,)
        assert grid[-1] == pytest.approx(1.0)
        assert grid[0] == pytest.approx(1.0 / num_thresholds)

    @pytest.mark.parametrize('num_thresholds', [0, -1])
    def test_rejects_empty_grid(self, num_thresholds):
        with pytest.raises(ValueError, match='at least 1'):
            threshold_grid(num_thresholds)


@pytest.mark.unittest
class TestPerTagSearch:
    def test_matches_reference(self):
        samples, labels = make_case()
        assert_all_identical(reference_optimal_thresholds(samples, labels),
                             compute_optimal_thresholds(samples, labels))

    def test_matches_reference_on_degenerate_tags(self):
        samples, labels = make_degenerate_case()
        assert_all_identical(reference_optimal_thresholds(samples, labels),
                             compute_optimal_thresholds(samples, labels))

    @pytest.mark.parametrize('num_thresholds', [4, 20, 100])
    def test_matches_reference_across_grid_sizes(self, num_thresholds):
        samples, labels = make_case(sample_num=384, tag_num=24, seed=num_thresholds)
        assert_all_identical(
            reference_optimal_thresholds(samples, labels, num_thresholds=num_thresholds),
            compute_optimal_thresholds(samples, labels, num_thresholds=num_thresholds),
        )

    @pytest.mark.parametrize('alpha', [0.5, 1.0, 2.0])
    def test_matches_reference_across_alpha(self, alpha):
        samples, labels = make_case(sample_num=384, tag_num=24, seed=3)
        assert_all_identical(reference_optimal_thresholds(samples, labels, alpha=alpha),
                             compute_optimal_thresholds(samples, labels, alpha=alpha))

    def test_single_tag(self):
        samples, labels = make_case(sample_num=256, tag_num=1, seed=5)
        assert_all_identical(reference_optimal_thresholds(samples, labels),
                             compute_optimal_thresholds(samples, labels))

    def test_max_workers_is_accepted_and_ignored(self):
        samples, labels = make_case(sample_num=256, tag_num=8)
        assert_all_identical(compute_optimal_thresholds(samples, labels),
                             compute_optimal_thresholds(samples, labels, max_workers=32))

    def test_requires_data_or_histograms(self):
        with pytest.raises(ValueError, match='histograms'):
            compute_optimal_thresholds()


@pytest.mark.unittest
class TestPerCategorySearch:
    def test_matches_reference(self):
        samples, labels = make_case()
        df_tags = category_frame(samples.shape[1])
        expected = reference_optimal_thresholds_by_categories(samples, labels, df_tags)
        actual = compute_optimal_thresholds_by_categories(samples, labels, df_tags)
        assert set(expected[0]) == set(actual[0])
        for exp, act in zip(expected, actual):
            for key in exp:
                assert exp[key] == act[key], f'category {key}: {exp[key]!r} != {act[key]!r}'

    def test_matches_reference_on_degenerate_tags(self):
        samples, labels = make_degenerate_case()
        df_tags = category_frame(samples.shape[1], groups=(0, 4))
        expected = reference_optimal_thresholds_by_categories(samples, labels, df_tags)
        actual = compute_optimal_thresholds_by_categories(samples, labels, df_tags)
        for exp, act in zip(expected, actual):
            for key in exp:
                assert exp[key] == act[key], f'category {key}: {exp[key]!r} != {act[key]!r}'

    def test_single_category(self):
        samples, labels = make_case(sample_num=256, tag_num=12, seed=9)
        df_tags = pd.DataFrame({'category': np.zeros(12, dtype=int)})
        expected = reference_optimal_thresholds_by_categories(samples, labels, df_tags)
        actual = compute_optimal_thresholds_by_categories(samples, labels, df_tags)
        assert expected == actual

    def test_requires_df_tags(self):
        samples, labels = make_case(sample_num=64, tag_num=4)
        with pytest.raises(ValueError, match='df_tags'):
            compute_optimal_thresholds_by_categories(samples, labels)

    def test_rejects_mismatched_tag_table(self):
        samples, labels = make_case(sample_num=64, tag_num=4)
        with pytest.raises(ValueError, match='histograms cover'):
            compute_optimal_thresholds_by_categories(samples, labels, category_frame(5))


@pytest.mark.unittest
class TestOfflineHistograms:
    def test_counts_every_sample_once(self):
        samples, labels = make_case(sample_num=300, tag_num=16)
        hist_all, hist_pos = build_threshold_histograms(samples, labels)
        assert hist_all.shape == (16, 101)
        np.testing.assert_array_equal(hist_all.sum(axis=1), np.full(16, 300.0))
        np.testing.assert_array_equal(hist_pos.sum(axis=1),
                                      labels.numpy().sum(axis=0).astype(np.float64))

    def test_positives_never_exceed_totals(self):
        samples, labels = make_case(sample_num=300, tag_num=16, seed=2)
        hist_all, hist_pos = build_threshold_histograms(samples, labels)
        assert (hist_pos <= hist_all).all()

    @pytest.mark.parametrize('chunk', [1, 3, 7, 16, 64])
    def test_chunking_is_transparent(self, chunk):
        samples, labels = make_case(sample_num=200, tag_num=16, seed=4)
        expected = build_threshold_histograms(samples, labels, chunk=1024)
        actual = build_threshold_histograms(samples, labels, chunk=chunk)
        np.testing.assert_array_equal(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])

    def test_reused_histograms_match_direct_call(self):
        samples, labels = make_case(sample_num=256, tag_num=20, seed=6)
        df_tags = category_frame(20)
        histograms = build_threshold_histograms(samples, labels)
        assert_all_identical(compute_optimal_thresholds(samples, labels),
                             compute_optimal_thresholds(histograms=histograms))
        assert (compute_optimal_thresholds_by_categories(samples, labels, df_tags)
                == compute_optimal_thresholds_by_categories(df_tags=df_tags, histograms=histograms))

    def test_rejects_shape_mismatch(self):
        samples, labels = make_case(sample_num=64, tag_num=4)
        with pytest.raises(ValueError, match='does not match'):
            build_threshold_histograms(samples, labels[:, :2])

    def test_rejects_non_2d_input(self):
        samples, labels = make_case(sample_num=64, tag_num=4)
        with pytest.raises(ValueError, match='got 1 dimensions'):
            build_threshold_histograms(samples[0], labels[0])

    def test_rejects_histograms_built_for_a_different_grid(self):
        """Indexing a 100-point grid with a 50-bin argmax would not raise on its own, it
        would just report thresholds off the wrong grid."""
        samples, labels = make_case(sample_num=128, tag_num=6, seed=25)
        coarse = build_threshold_histograms(samples, labels, num_thresholds=50)
        with pytest.raises(ValueError, match='num_thresholds is 100'):
            compute_optimal_thresholds(histograms=coarse)
        with pytest.raises(ValueError, match='num_thresholds is 100'):
            compute_optimal_thresholds_by_categories(df_tags=category_frame(6), histograms=coarse)
        # ... and it works when the caller says which grid the histograms belong to
        assert_all_identical(
            reference_optimal_thresholds(samples, labels, num_thresholds=50),
            compute_optimal_thresholds(histograms=coarse, num_thresholds=50),
        )

    def test_rejects_mismatched_histogram_pair(self):
        samples, labels = make_case(sample_num=128, tag_num=6, seed=26)
        hist_all, hist_pos = build_threshold_histograms(samples, labels)
        with pytest.raises(ValueError, match='shapes disagree'):
            compute_optimal_thresholds(histograms=(hist_all, hist_pos[:3]))


@pytest.mark.unittest
class TestStreamingHistogram:
    @pytest.mark.parametrize('batch_size', [1, 5, 32, 128, 1000])
    def test_matches_offline_for_any_batching(self, batch_size):
        samples, labels = make_case(sample_num=257, tag_num=16, seed=8)
        expected = build_threshold_histograms(samples, labels)

        accumulator = StreamingThresholdHistogram(16)
        for start in range(0, samples.shape[0], batch_size):
            accumulator.update(samples[start:start + batch_size], labels[start:start + batch_size])
        actual = accumulator.finalize()

        np.testing.assert_array_equal(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])

    def test_end_to_end_matches_reference(self):
        samples, labels = make_case(sample_num=384, tag_num=24, seed=10)
        df_tags = category_frame(24)

        accumulator = StreamingThresholdHistogram(24)
        for start in range(0, samples.shape[0], 37):
            accumulator.update(samples[start:start + 37], labels[start:start + 37])
        histograms = accumulator.finalize()

        assert_all_identical(reference_optimal_thresholds(samples, labels),
                             compute_optimal_thresholds(histograms=histograms))
        expected = reference_optimal_thresholds_by_categories(samples, labels, df_tags)
        actual = compute_optimal_thresholds_by_categories(df_tags=df_tags, histograms=histograms)
        assert expected == actual

    def test_works_without_any_update(self):
        """An empty shard must still produce well-formed, all-zero histograms."""
        hist_all, hist_pos = StreamingThresholdHistogram(6).finalize()
        assert hist_all.shape == (6, 101)
        assert not hist_all.any()
        assert not hist_pos.any()

    def test_defaults_to_cpu_without_cuda(self):
        """CPU-only environments must work with no device argument at all."""
        samples, labels = make_case(sample_num=64, tag_num=4, seed=12)
        accumulator = StreamingThresholdHistogram(4)
        accumulator.update(samples, labels)
        assert accumulator.hist_all.device.type == 'cpu'
        expected = build_threshold_histograms(samples, labels)
        np.testing.assert_array_equal(expected[0], accumulator.finalize()[0])

    def test_explicit_cpu_device(self):
        samples, labels = make_case(sample_num=64, tag_num=4, seed=13)
        accumulator = StreamingThresholdHistogram(4, device='cpu')
        accumulator.update(samples, labels)
        np.testing.assert_array_equal(build_threshold_histograms(samples, labels)[0],
                                      accumulator.finalize()[0])

    def test_finalize_without_accelerator_is_identity(self):
        samples, labels = make_case(sample_num=64, tag_num=4, seed=14)
        accumulator = StreamingThresholdHistogram(4)
        accumulator.update(samples, labels)
        np.testing.assert_array_equal(accumulator.finalize()[0], accumulator.finalize(None)[0])

    def test_finalize_skips_reduce_for_single_process(self):
        """accelerator.reduce must not be touched when there is nothing to reduce."""

        class _SingleProcessAccelerator:
            num_processes = 1

            def reduce(self, tensor, reduction='sum'):
                raise AssertionError('reduce must not be called for a single process')

        samples, labels = make_case(sample_num=64, tag_num=4, seed=15)
        accumulator = StreamingThresholdHistogram(4)
        accumulator.update(samples, labels)
        np.testing.assert_array_equal(build_threshold_histograms(samples, labels)[1],
                                      accumulator.finalize(_SingleProcessAccelerator())[1])

    def test_finalize_reduces_when_distributed(self):
        """Two shards reduced with sum must equal one pass over the concatenation."""

        class _SummingAccelerator:
            """Stands in for a 2-process run whose peer holds `other`."""

            num_processes = 2

            def __init__(self, peer):
                self.peer = peer
                self._calls = 0

            def reduce(self, tensor, reduction='sum'):
                assert reduction == 'sum'
                peer_tensor = (self.peer.hist_all, self.peer.hist_pos)[self._calls]
                self._calls += 1
                return tensor + peer_tensor

        samples, labels = make_case(sample_num=200, tag_num=10, seed=16)
        first, second = slice(0, 120), slice(120, 200)

        shard_a = StreamingThresholdHistogram(10)
        shard_a.update(samples[first], labels[first])
        shard_b = StreamingThresholdHistogram(10)
        shard_b.update(samples[second], labels[second])

        reduced = shard_a.finalize(_SummingAccelerator(shard_b))
        expected = build_threshold_histograms(samples, labels)
        np.testing.assert_array_equal(expected[0], reduced[0])
        np.testing.assert_array_equal(expected[1], reduced[1])

    def test_rejects_shape_mismatch(self):
        samples, labels = make_case(sample_num=32, tag_num=4)
        accumulator = StreamingThresholdHistogram(4)
        with pytest.raises(ValueError, match='does not match'):
            accumulator.update(samples, labels[:, :2])

    def test_rejects_wrong_tag_count(self):
        samples, labels = make_case(sample_num=32, tag_num=4)
        accumulator = StreamingThresholdHistogram(5)
        with pytest.raises(ValueError, match='Expected 5 tags'):
            accumulator.update(samples, labels)

    def test_rejects_non_2d_batch(self):
        samples, labels = make_case(sample_num=32, tag_num=4)
        accumulator = StreamingThresholdHistogram(4)
        with pytest.raises(ValueError, match='got 1 dimensions'):
            accumulator.update(samples[0], labels[0])


@pytest.mark.unittest
class TestScoreDtypes:
    @pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
    def test_matches_reference(self, dtype):
        samples, labels = make_case(sample_num=256, tag_num=12, seed=23)
        samples, labels = samples.to(dtype), labels.to(dtype)
        assert_all_identical(reference_optimal_thresholds(samples, labels),
                             compute_optimal_thresholds(samples, labels))

    @pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
    def test_streaming_matches_offline(self, dtype):
        samples, labels = make_case(sample_num=256, tag_num=12, seed=24)
        samples, labels = samples.to(dtype), labels.to(dtype)
        accumulator = StreamingThresholdHistogram(12)
        for start in range(0, samples.shape[0], 48):
            accumulator.update(samples[start:start + 48], labels[start:start + 48])
        np.testing.assert_array_equal(build_threshold_histograms(samples, labels)[0],
                                      accumulator.finalize()[0])

    @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
    def test_half_precision_scores_are_promoted(self, dtype):
        """Mixed-precision inference can hand back half-precision scores. A 0.01 grid is
        nowhere near representable in half, so the comparison is promoted; numpy cannot
        even hold bfloat16, so the old code would have failed outright here."""
        scores = [0.05, 0.15, 0.35, 0.55, 0.95]
        labels = [0.0, 0.0, 1.0, 1.0, 1.0]
        half = torch.tensor([[s] for s in scores], dtype=dtype)
        half_labels = torch.tensor([[v] for v in labels], dtype=dtype)

        hist_all, hist_pos = build_threshold_histograms(half, half_labels, num_thresholds=10)
        # same bins as the float32 hand-worked example: rounding to half moves each score
        # by less than the distance to its neighbouring threshold
        assert {i: int(v) for i, v in enumerate(hist_all[0]) if v} == {0: 1, 1: 1, 3: 1, 5: 1, 9: 1}
        assert {i: int(v) for i, v in enumerate(hist_pos[0]) if v} == {3: 1, 5: 1, 9: 1}

        th, f1, _, _ = compute_optimal_thresholds(half, half_labels, num_thresholds=10)
        assert th[0] == pytest.approx(0.25)
        assert f1[0] == pytest.approx(1.0)

        # and the streaming path agrees with the offline one on the same input
        accumulator = StreamingThresholdHistogram(1, num_thresholds=10)
        accumulator.update(half, half_labels)
        np.testing.assert_array_equal(hist_all, accumulator.finalize()[0])

    def test_half_promotion_changes_a_boundary_score(self):
        """Pins the promotion down with a score the precisions disagree about.

        float16 rounds 0.1 *down* to 0.099975..., below the true 0.1 threshold, so a score
        sitting exactly on the half representation of the first threshold falls below it
        once promoted, and lands in bin 0. Comparing in half would put it in bin 1.
        """
        boundary = float(np.float16(0.1))
        assert boundary < 0.1

        scores = torch.tensor([[boundary]], dtype=torch.float16)
        labels = torch.ones((1, 1), dtype=torch.float16)
        hist_all, _ = build_threshold_histograms(scores, labels, num_thresholds=10)
        assert int(np.argmax(hist_all[0])) == 0

        accumulator = StreamingThresholdHistogram(1, num_thresholds=10)
        accumulator.update(scores, labels)
        assert int(np.argmax(accumulator.finalize()[0][0])) == 0

    def test_float32_scores_are_judged_against_unrounded_thresholds(self):
        """The thresholds are float64 and the original compared `sample >= th` in numpy,
        which promotes the float32 score rather than rounding the threshold. Bucketizing
        in float32 would round the threshold instead and flip scores within one ulp of it:
        float32(0.03) is 0.0299999993, below the real 0.03, yet not below float32(0.03).
        """
        grid = threshold_grid(100)
        score = np.float32(grid[2])                   # the float32 image of the 0.03 edge
        assert float(score) < grid[2]                 # genuinely below the real threshold
        assert score >= np.float32(grid[2])           # but not below its float32 rounding

        samples = torch.tensor([[float(score)]], dtype=torch.float32)
        hist_all, _ = build_threshold_histograms(samples, torch.ones_like(samples))
        assert int(np.argmax(hist_all[0])) == 2       # 0.01 and 0.02 only, not 0.03

    def test_matches_original_numpy_semantics_on_every_threshold_boundary(self):
        """Sweep four float32 steps either side of every threshold and require the bins to
        agree with `score.astype(float64) >= thresholds`, which is what the original scan
        computed."""
        grid = threshold_grid(100)
        edge = []
        for th in grid:
            v = np.float32(th)
            for _ in range(4):
                edge.append(v)
                v = np.nextafter(v, np.float32(0))
            v = np.float32(th)
            for _ in range(4):
                v = np.nextafter(v, np.float32(2))
                edge.append(v)
        edge = np.array(edge, dtype=np.float32)
        expected = (edge[:, None].astype(np.float64) >= grid[None, :]).sum(axis=1)

        # one score per tag, so each row's histogram has a single entry and its argmax is
        # exactly that score's bin
        samples = torch.from_numpy(edge.reshape(1, -1))
        hist_all, _ = build_threshold_histograms(samples, torch.ones_like(samples))
        assert hist_all.shape == (edge.size, 101)
        np.testing.assert_array_equal(hist_all.sum(axis=1), np.ones(edge.size))
        np.testing.assert_array_equal(expected, hist_all.argmax(axis=1))


@pytest.mark.unittest
class TestHistogramCountsAreExact:
    def test_positive_counts_are_integral(self):
        """hist_pos is built by counting bins, not by float-weighted accumulation, so
        the counts stay exact no matter how many samples land in one bin."""
        sample_num = 5000
        samples = torch.full((sample_num, 2), 0.5, dtype=torch.float32)
        labels = torch.ones((sample_num, 2), dtype=torch.float32)
        hist_all, hist_pos = build_threshold_histograms(samples, labels)
        np.testing.assert_array_equal(hist_all, hist_pos)
        assert hist_pos.sum() == 2 * sample_num
        assert (hist_pos == np.floor(hist_pos)).all()


@pytest.mark.unittest
class TestLabelConvention:
    def test_only_exact_one_counts_as_positive(self):
        """Preserves the original `.to(int32).to(bool)` semantics: a soft label below
        1.0 truncates to zero and is treated as negative."""
        samples = torch.tensor([[0.9], [0.9], [0.9]], dtype=torch.float32)
        labels = torch.tensor([[1.0], [0.99], [0.4]], dtype=torch.float32)
        _, hist_pos = build_threshold_histograms(samples, labels)
        assert hist_pos.sum() == 1.0

    def test_matches_reference_with_soft_labels(self):
        rng = np.random.default_rng(17)
        samples = torch.from_numpy(rng.random((128, 6)).astype(np.float32))
        labels = torch.from_numpy(rng.random((128, 6)).astype(np.float32))
        assert_all_identical(reference_optimal_thresholds(samples, labels),
                             compute_optimal_thresholds(samples, labels))


@pytest.mark.unittest
class TestAcceleratorIntegration:
    """`test.py` drives the accumulator through an `Accelerator`, including on machines
    with no GPU at all, so exercise that path with a real one."""

    def test_cpu_only_accelerator_round_trip(self):
        accelerate = pytest.importorskip('accelerate')
        accelerator = accelerate.Accelerator(cpu=True)
        assert accelerator.device.type == 'cpu'
        assert accelerator.num_processes == 1

        samples, labels = make_case(sample_num=192, tag_num=12, seed=22)
        accumulator = StreamingThresholdHistogram(12, device=accelerator.device)
        for start in range(0, samples.shape[0], 32):
            accumulator.update(samples[start:start + 32].to(accelerator.device),
                               labels[start:start + 32].to(accelerator.device))
        histograms = accumulator.finalize(accelerator=accelerator)

        expected = build_threshold_histograms(samples, labels)
        np.testing.assert_array_equal(expected[0], histograms[0])
        np.testing.assert_array_equal(expected[1], histograms[1])
        assert_all_identical(reference_optimal_thresholds(samples, labels),
                             compute_optimal_thresholds(histograms=histograms))


@pytest.mark.unittest
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
class TestCudaParity:
    def test_offline_histograms_match_cpu(self):
        samples, labels = make_case(sample_num=300, tag_num=20, seed=18)
        expected = build_threshold_histograms(samples, labels, device='cpu')
        actual = build_threshold_histograms(samples, labels, device='cuda')
        np.testing.assert_array_equal(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])

    def test_streaming_matches_cpu(self):
        samples, labels = make_case(sample_num=300, tag_num=20, seed=19)
        expected = build_threshold_histograms(samples, labels, device='cpu')

        accumulator = StreamingThresholdHistogram(20, device='cuda')
        for start in range(0, samples.shape[0], 64):
            accumulator.update(samples[start:start + 64].cuda(), labels[start:start + 64].cuda())
        actual = accumulator.finalize()

        np.testing.assert_array_equal(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])

    def test_streaming_accepts_cpu_batches_on_cuda_accumulator(self):
        samples, labels = make_case(sample_num=128, tag_num=10, seed=20)
        accumulator = StreamingThresholdHistogram(10, device='cuda')
        accumulator.update(samples, labels)
        np.testing.assert_array_equal(build_threshold_histograms(samples, labels)[0],
                                      accumulator.finalize()[0])

    def test_end_to_end_matches_reference_on_cuda(self):
        samples, labels = make_case(sample_num=256, tag_num=16, seed=21)
        assert_all_identical(
            reference_optimal_thresholds(samples, labels),
            compute_optimal_thresholds(samples.cuda(), labels.cuda(), device='cuda'),
        )
