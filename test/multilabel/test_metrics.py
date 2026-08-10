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

    def test_float64_input_compares_in_float64(self):
        """A score just below a threshold in float64, but indistinguishable from it once
        rounded to float32, must be judged the way the original float64 numpy comparison
        judged it: below."""
        grid = threshold_grid(100)
        edge = grid[6] - 1e-12                       # just under the 0.07 threshold
        assert edge < grid[6]                        # ... in float64
        assert np.float32(edge) >= np.float32(grid[6])  # ... but not in float32

        labels = torch.ones((1, 1), dtype=torch.float64)
        hist_all, _ = build_threshold_histograms(torch.tensor([[edge]], dtype=torch.float64), labels)
        # the bin counts thresholds <= score, so 0.07 must not be counted
        assert int(np.argmax(hist_all[0])) == 6

        # the same value handed in as float32 genuinely is >= the threshold
        hist32, _ = build_threshold_histograms(torch.tensor([[edge]], dtype=torch.float32),
                                               labels.to(torch.float32))
        assert int(np.argmax(hist32[0])) == 7


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
