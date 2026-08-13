import pytest
import torch

from animetimm.multilabel.losses import (AsymmetricLoss, BCELoss, LOSS_NAMES,
                                         PartialAsymmetricLoss, build_loss,
                                         neg_gamma_from_scores)


@pytest.fixture
def logits():
    torch.manual_seed(0)
    return torch.randn(8, 32) * 3


@pytest.fixture
def hard_targets():
    torch.manual_seed(1)
    return (torch.rand(8, 32) < 0.15).float()


@pytest.fixture
def soft_targets():
    torch.manual_seed(2)
    return torch.rand(8, 32)


@pytest.mark.unittest
class TestBCELoss:
    def test_matches_torch_exactly(self, logits, hard_targets):
        ref = torch.nn.BCEWithLogitsLoss(reduction='none')(logits, hard_targets)
        assert torch.equal(BCELoss()(logits, hard_targets), ref)

    def test_matches_torch_on_soft_targets(self, logits, soft_targets):
        # mixup is on by default in this codebase, so soft targets are the norm
        ref = torch.nn.BCEWithLogitsLoss(reduction='none')(logits, soft_targets)
        assert torch.equal(BCELoss()(logits, soft_targets), ref)

    def test_shape_is_unreduced(self, logits, hard_targets):
        assert BCELoss()(logits, hard_targets).shape == logits.shape


@pytest.mark.unittest
class TestAsymmetricLoss:
    def test_reduces_to_bce_when_neutral(self, logits, hard_targets):
        # no focusing and no probability shift is plain BCE
        got = AsymmetricLoss(gamma_neg=0, gamma_pos=0, clip=0)(logits, hard_targets)
        ref = torch.nn.BCEWithLogitsLoss(reduction='none')(logits, hard_targets)
        assert torch.allclose(got, ref, atol=1e-5)

    def test_reduces_to_bce_on_soft_targets(self, logits, soft_targets):
        got = AsymmetricLoss(gamma_neg=0, gamma_pos=0, clip=0)(logits, soft_targets)
        ref = torch.nn.BCEWithLogitsLoss(reduction='none')(logits, soft_targets)
        assert torch.allclose(got, ref, atol=1e-5)

    def test_non_negative(self, logits, hard_targets):
        assert (AsymmetricLoss()(logits, hard_targets) >= 0).all()

    def test_down_weights_easy_negatives(self):
        # the whole point: a negative the model already scores low should cost
        # far less under ASL than under BCE, while a positive should not
        z = torch.tensor([[-6.0, 6.0]])
        y = torch.tensor([[0.0, 1.0]])
        bce = BCELoss()(z, y)
        asl = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)(z, y)
        assert asl[0, 0] < bce[0, 0] * 0.01
        assert asl[0, 1] == pytest.approx(bce[0, 1].item(), rel=1e-4)

    def test_clip_zeroes_confident_negatives(self):
        # a negative scored below the margin is fully discounted; that is the
        # mechanism protecting against unlabeled positives
        z = torch.tensor([[-8.0]])
        y = torch.tensor([[0.0]])
        assert AsymmetricLoss(clip=0.05)(z, y).item() == pytest.approx(0.0, abs=1e-6)

    def test_gradient_flows_and_is_finite(self, logits, hard_targets):
        z = logits.clone().requires_grad_(True)
        AsymmetricLoss()(z, hard_targets).sum().backward()
        assert z.grad is not None and torch.isfinite(z.grad).all()

    def test_stable_at_saturation(self):
        z = torch.tensor([[-60.0, 60.0, 0.0]], requires_grad=True)
        y = torch.tensor([[1.0, 0.0, 1.0]])
        out = AsymmetricLoss()(z, y)
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert torch.isfinite(z.grad).all()

    def test_detach_focus_changes_gradient_only(self, logits, hard_targets):
        a = AsymmetricLoss(detach_focus=True)(logits, hard_targets)
        b = AsymmetricLoss(detach_focus=False)(logits, hard_targets)
        assert torch.allclose(a, b, atol=1e-6)


@pytest.mark.unittest
class TestPartialAsymmetricLoss:
    def test_matches_asl_without_extras(self, logits, hard_targets):
        a = PartialAsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)(
            logits, hard_targets)
        b = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)(logits, hard_targets)
        assert torch.allclose(a, b, atol=1e-6)

    def test_per_tag_gamma_is_applied_per_column(self, hard_targets):
        c = hard_targets.shape[1]
        z = torch.full((8, c), -2.0)
        y = torch.zeros(8, c)
        g = torch.cat([torch.full((c // 2,), 0.0), torch.full((c - c // 2,), 6.0)])
        out = PartialAsymmetricLoss(neg_gamma=g, clip=0.0)(z, y)
        # higher gamma on the right half must suppress those negatives
        assert out[:, :c // 2].mean() > out[:, c // 2:].mean() * 5

    def test_topk_gate_drops_exactly_k_negatives_per_row(self):
        torch.manual_seed(3)
        z = torch.randn(4, 20) * 2
        y = torch.zeros(4, 20)
        out = PartialAsymmetricLoss(gamma_neg=0, clip=0.0, ignore_topk=5)(z, y)
        assert (out == 0).sum(dim=1).tolist() == [5, 5, 5, 5]

    def test_topk_gate_drops_the_highest_scoring_negatives(self):
        z = torch.tensor([[5.0, 4.0, -1.0, -2.0, -3.0]])
        y = torch.zeros(1, 5)
        out = PartialAsymmetricLoss(gamma_neg=0, clip=0.0, ignore_topk=2)(z, y)
        assert out[0, 0] == 0 and out[0, 1] == 0
        assert (out[0, 2:] > 0).all()

    def test_topk_gate_never_touches_positives(self):
        z = torch.tensor([[9.0, 8.0, 7.0, -5.0]])
        y = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        out = PartialAsymmetricLoss(gamma_neg=0, clip=0.0, ignore_topk=3)(z, y)
        assert (out[0, :2] > 0).all(), 'positives must survive the gate'

    def test_topk_zero_is_a_no_op(self, logits, hard_targets):
        a = PartialAsymmetricLoss(ignore_topk=0)(logits, hard_targets)
        b = PartialAsymmetricLoss()(logits, hard_targets)
        assert torch.equal(a, b)

    def test_diligence_gates_sparse_samples_harder(self):
        # two rows, same logits; the first carries few labels, the second many
        z = torch.full((2, 40), 1.0)
        y = torch.zeros(2, 40)
        y[0, :4] = 1.0
        y[1, :30] = 1.0
        loss = PartialAsymmetricLoss(gamma_neg=0, clip=0.0, ignore_topk=4,
                                     diligence_modulate=True, diligence_lo=8,
                                     diligence_hi=32, diligence_max_scale=3.0)
        out = loss(z, y)
        dropped = [(out[i] == 0).sum().item() - int(y[i].sum().item() == 0) for i in range(2)]
        n0 = ((out[0] == 0) & (y[0] == 0)).sum().item()
        n1 = ((out[1] == 0) & (y[1] == 0)).sum().item()
        assert n0 > n1, f'sparsely-tagged row should lose more negatives, got {n0} vs {n1}'

    def test_diligence_off_gives_equal_gating(self):
        z = torch.full((2, 40), 1.0)
        y = torch.zeros(2, 40)
        y[0, :4] = 1.0
        y[1, :30] = 1.0
        out = PartialAsymmetricLoss(gamma_neg=0, clip=0.0, ignore_topk=4)(z, y)
        n0 = ((out[0] == 0) & (y[0] == 0)).sum().item()
        n1 = ((out[1] == 0) & (y[1] == 0)).sum().item()
        assert n0 == n1 == 4

    def test_gradient_finite_with_all_mechanisms(self, hard_targets):
        z = (torch.randn(8, 32) * 4).requires_grad_(True)
        g = neg_gamma_from_scores(torch.rand(32), gamma_neg=2, gamma_unann=7)
        loss = PartialAsymmetricLoss(neg_gamma=g, ignore_topk=6,
                                     diligence_modulate=True)
        out = loss(z, hard_targets)
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert torch.isfinite(z.grad).all()

    def test_soft_targets_supported(self, logits, soft_targets):
        out = PartialAsymmetricLoss(ignore_topk=4)(logits, soft_targets)
        assert out.shape == logits.shape and torch.isfinite(out).all()


@pytest.mark.unittest
class TestNegGammaFromScores:
    def test_endpoints_and_direction(self):
        s = torch.linspace(0, 1, 101)
        g = neg_gamma_from_scores(s, gamma_neg=2.0, gamma_unann=7.0)
        assert g[0].item() == pytest.approx(7.0, abs=1e-4), 'least reliable -> gamma_unann'
        assert g[-1].item() == pytest.approx(2.0, abs=1e-4), 'most reliable -> gamma_neg'
        assert (g[1:] <= g[:-1] + 1e-6).all(), 'must be monotone decreasing'

    def test_outliers_are_clamped_not_extrapolated(self):
        s = torch.tensor([-100.0, 0.4, 0.5, 0.6, 100.0])
        g = neg_gamma_from_scores(s, gamma_neg=2.0, gamma_unann=7.0)
        assert g.min() >= 2.0 - 1e-6 and g.max() <= 7.0 + 1e-6

    def test_nan_scores_fall_back_to_reliable(self):
        s = torch.tensor([float('nan'), 0.0, 1.0])
        g = neg_gamma_from_scores(s, gamma_neg=2.0, gamma_unann=7.0)
        assert torch.isfinite(g).all()
        assert g[0].item() == pytest.approx(2.0, abs=1e-4)


@pytest.mark.unittest
class TestBuildLoss:
    @pytest.mark.parametrize('name', LOSS_NAMES)
    def test_every_name_builds_and_runs(self, name, logits, hard_targets):
        out = build_loss(name)(logits, hard_targets)
        assert out.shape == logits.shape and torch.isfinite(out).all()

    def test_default_is_bce(self, logits, hard_targets):
        ref = torch.nn.BCEWithLogitsLoss(reduction='none')(logits, hard_targets)
        assert torch.equal(build_loss()(logits, hard_targets), ref)

    def test_unknown_name_rejected(self):
        with pytest.raises(ValueError, match='Unknown loss'):
            build_loss('nope')


@pytest.mark.unittest
class TestTrainIntegration:
    """The two prior entry points and the promise that `bce` changes nothing."""

    @staticmethod
    def _tags_info(df):
        from animetimm.multilabel.dataset import TagsInfo
        return TagsInfo(df=df, tags_to_id={t: i for i, t in enumerate(df['name'])},
                        tags=df['name'].tolist(),
                        weights=df.get('weights', 1.0))

    @pytest.fixture
    def tags_df(self):
        import pandas as pd
        return pd.DataFrame({'name': [f't{i}' for i in range(6)],
                             'category': [0] * 6,
                             'reliability': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]})

    def _make(self, tags_df, **kw):
        from animetimm.multilabel.train import _make_loss_fn
        base = dict(loss='pasl', tags_info=self._tags_info(tags_df), gamma_neg=2.0,
                    gamma_pos=0.0, gamma_unann=7.0, clip=0.05, prior_file=None,
                    prior_column='reliability', ignore_topk=0, diligence=False,
                    diligence_lo=12.0, diligence_hi=48.0, diligence_max_scale=2.0)
        base.update(kw)
        return _make_loss_fn(**base)

    def test_bce_arm_is_the_untouched_torch_loss(self, tags_df, logits, hard_targets):
        fn = self._make(tags_df, loss='bce')
        assert isinstance(fn, torch.nn.BCEWithLogitsLoss)
        ref = torch.nn.BCEWithLogitsLoss(reduction='none')(logits, hard_targets)
        assert torch.equal(fn(logits, hard_targets), ref)

    def test_prior_read_from_dataset_column(self, tags_df):
        fn = self._make(tags_df)
        assert fn.neg_gamma is not None
        assert fn.neg_gamma[0] > fn.neg_gamma[-1], 'unreliable tag must focus harder'

    def test_missing_column_falls_back_to_flat_gamma(self, tags_df):
        fn = self._make(tags_df.drop(columns=['reliability']))
        assert fn.neg_gamma is None

    def test_file_overrides_dataset_column(self, tags_df, tmp_path):
        import pandas as pd
        # reversed reliability in the file; if the file wins, so does its order
        p = tmp_path / 'prior.parquet'
        pd.DataFrame({'name': tags_df['name'],
                      'reliability': tags_df['reliability'][::-1].to_numpy()}
                     ).to_parquet(p)
        fn = self._make(tags_df, prior_file=str(p))
        assert fn.neg_gamma[0] < fn.neg_gamma[-1]

    def test_file_missing_column_raises(self, tags_df, tmp_path):
        import pandas as pd
        p = tmp_path / 'bad.parquet'
        pd.DataFrame({'name': tags_df['name'], 'other': 1.0}).to_parquet(p)
        with pytest.raises(KeyError, match='reliability'):
            self._make(tags_df, prior_file=str(p))

    def test_tags_absent_from_file_keep_reliable_gamma(self, tags_df, tmp_path):
        import pandas as pd
        p = tmp_path / 'partial.parquet'
        pd.DataFrame({'name': ['t0'], 'reliability': [0.0]}).to_parquet(p)
        fn = self._make(tags_df, prior_file=str(p))
        # t0 is the only scored tag and it is the least reliable; the unscored
        # ones must not be softened
        assert fn.neg_gamma[0] == pytest.approx(7.0, abs=1e-4)
        assert all(g == pytest.approx(2.0, abs=1e-4) for g in fn.neg_gamma[1:])
