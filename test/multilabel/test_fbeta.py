import numpy as np
import pytest
import torch

from animetimm.multilabel.metrics import (_cat_beta, _curves, f1score,
                                          compute_optimal_thresholds)


@pytest.mark.unittest
class TestFBetaConvention:
    """`alpha` used to mean beta-squared in f1score and beta in _curves.  Both
    now mean beta; these tests pin that down so it cannot drift apart again."""

    def test_f1score_beta1_is_plain_f1(self):
        tp, fp, tn, fn = (torch.tensor([30.0]), torch.tensor([10.0]),
                          torch.tensor([100.0]), torch.tensor([20.0]))
        got = float(f1score(tp, fp, tn, fn))
        assert got == pytest.approx(2 * 30 / (2 * 30 + 10 + 20), rel=1e-6)  # float32

    @pytest.mark.parametrize('beta', [0.5, 1.0, 2.0, 3.0])
    def test_f1score_matches_the_textbook_formula(self, beta):
        tp, fp, tn, fn = (torch.tensor([30.0]), torch.tensor([10.0]),
                          torch.tensor([100.0]), torch.tensor([20.0]))
        p, r = 30 / 40, 30 / 50
        want = (1 + beta ** 2) * p * r / (beta ** 2 * p + r)
        assert float(f1score(tp, fp, tn, fn, beta=beta)) == pytest.approx(want, rel=1e-6)

    def test_alpha_alias_now_agrees_with_beta(self):
        tp, fp, tn, fn = (torch.tensor([30.0]), torch.tensor([10.0]),
                          torch.tensor([100.0]), torch.tensor([20.0]))
        assert float(f1score(tp, fp, tn, fn, alpha=2.0)) == \
               pytest.approx(float(f1score(tp, fp, tn, fn, beta=2.0)), rel=1e-12)

    def test_curves_and_f1score_agree_on_the_same_confusion(self):
        # one tag, scores in two bins: 3 positives above, 1 below; 2 negatives above
        hist_pos = np.array([[1.0, 3.0]])
        hist_all = np.array([[1.0, 5.0]])
        for beta in (1.0, 2.0):
            f, p, r = _curves(hist_all, hist_pos, beta)
            tp, fp, fn = 3.0, 2.0, 1.0
            want = float(f1score(torch.tensor([tp]), torch.tensor([fp]),
                                 torch.tensor([0.0]), torch.tensor([fn]), beta=beta))
            assert f[0, 0] == pytest.approx(want, rel=1e-6), f'beta={beta}'

    def test_beta_above_one_favours_recall(self):
        tp, fp, tn, fn = (torch.tensor([30.0]), torch.tensor([10.0]),
                          torch.tensor([0.0]), torch.tensor([20.0]))
        # recall 0.60 < precision 0.75, so weighting recall must lower the score
        assert float(f1score(tp, fp, tn, fn, beta=2.0)) < float(f1score(tp, fp, tn, fn))


@pytest.mark.unittest
class TestPerTagBeta:
    def test_curves_accepts_a_per_tag_vector(self):
        hist_pos = np.array([[1.0, 3.0], [1.0, 3.0]])
        hist_all = np.array([[1.0, 5.0], [1.0, 5.0]])
        f, _, _ = _curves(hist_all, hist_pos, np.array([1.0, 3.0]))
        fa, _, _ = _curves(hist_all[:1], hist_pos[:1], 1.0)
        fb, _, _ = _curves(hist_all[:1], hist_pos[:1], 3.0)
        assert f[0, 0] == pytest.approx(fa[0, 0], rel=1e-6)  # float32
        assert f[1, 0] == pytest.approx(fb[0, 0], rel=1e-6)  # float32

    def test_scalar_and_uniform_vector_agree(self):
        hist_pos = np.array([[1.0, 3.0], [2.0, 2.0]])
        hist_all = np.array([[1.0, 5.0], [4.0, 6.0]])
        a, _, _ = _curves(hist_all, hist_pos, 2.0)
        b, _, _ = _curves(hist_all, hist_pos, np.array([2.0, 2.0]))
        assert np.allclose(a, b)

    def test_higher_beta_never_raises_the_chosen_threshold(self):
        rng = np.random.default_rng(0)
        n = 20_000
        y = (rng.random(n) < 0.05)
        s = 1 / (1 + np.exp(-rng.normal(np.where(y, 1.5, 0.0), 1.0)))
        smp = torch.tensor(s, dtype=torch.float32)[:, None]
        lbl = torch.tensor(y, dtype=torch.float32)[:, None]
        prev = 1.1
        for beta in (1.0, 1.5, 2.0, 3.0):
            th = compute_optimal_thresholds(all_sample=smp, all_labels=lbl,
                                            beta=beta, num_thresholds=100)[0][0]
            assert th <= prev + 1e-6, f'beta={beta} raised the threshold'
            prev = th

    def test_cat_beta_reduces_a_vector_by_mean(self):
        assert _cat_beta(np.array([1.0, 2.0, 3.0, 9.0]),
                         np.array([True, True, False, False])) == pytest.approx(1.5)
        assert _cat_beta(2.5, np.array([True, False])) == pytest.approx(2.5)
