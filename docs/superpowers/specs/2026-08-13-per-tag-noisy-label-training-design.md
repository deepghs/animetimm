# Per-tag noise-aware training for animetimm

2026-08-13. Design and experiment record for differentiated, per-tag handling of
label noise in dbv4 training.

## 1 The problem, stated precisely

Danbooru tag data is high-precision and low-recall. The
[label-quality survey](https://github.com/deepghs-research/danbooru-label-quality-survey)
measured, by human eye on 150 images, general-tag precision of 96.1% and
checklist recall of 49.8%, and showed the recall varies threefold between
semantic families — object/text/count at 81.9% against composition/scene/pose at
25.4%. Training with plain BCE therefore feeds the model a large and
*systematically structured* stream of false negatives.

Formally the label channel per tag `t` has two noise rates:

```
rho_plus_t  = P(GT = 0 | dbv4 = 1) = 1 - precision_t     measured at ~3.6%, flat across tags
rho_minus_t = P(GT = 1 | dbv4 = 0)                        varies enormously
```

`rho_minus` is the actionable one, and it is what the losses below target. Note
this is not `1 - recall`: the survey's follow-up analysis
([covariates.md](https://github.com/deepghs-research/danbooru-label-quality-survey/blob/master/covariates.md) §2)
shows `rho_minus = (1 - r) * pi / (1 - a)`, so it rises with prevalence even at
constant recall.

## 2 Why the shipped per-tag metrics cannot set the softening amount

The obvious idea is to read the per-tag metrics animetimm already publishes in
`selected_tags.csv`. Measured against the survey's validated per-tag omission
proxy, over 8,801 live general tags:

| metric | rho vs omission | controlling for frequency | controlling for frequency + `test_mcc` |
|---|---:|---:|---:|
| `test_precision` | −0.352 | −0.339 | **−0.043** |
| `test_recall` | −0.346 | −0.330 | **+0.061** |
| `test_f1` | −0.358 | −0.343 | **+0.027** |
| `best_threshold` | −0.204 | −0.182 | **−0.018** |
| `log(P/R)` | +0.256 | +0.232 | **−0.066** |

The whole CSV is **one degree of freedom**. Six models of 100M–1013M parameters
across CNN and ViT families agree on per-tag `test_mcc` at Spearman 0.954–0.977;
PC1 explains 97.2% of the between-tag variance with near-identical loadings. So
per-tag difficulty is a property of the data, not of any model — but that single
factor fuses label noise with intrinsic difficulty, and the two need **opposite**
treatments. Softening a hard-but-cleanly-labelled tag throws away real
supervision.

Averaging the six models does not help: the shared factor correlates with
omission at −0.342, the same as any single model. Capacity and resolution
sensitivity, which might have separated "capacity-limited" from "noise-limited",
give only −0.102 and +0.105.

**Conclusion: use the CSV for difficulty, not for noise. The noise estimate has
to come from somewhere else.**

## 3 Where the noise estimate comes from

Two estimators, each used only where it has range, combined by within-category
rank. Built by `tools/build_tag_reliability.py`.

**Elkan-Noto.** A model trained on positive-unlabeled data learns
`g(x) = P(labelled | x) = c * p(x)`; on the labelled-positive set `p(x) = 1`, so
`c = E[g(x) | labelled positive]` estimates the tag's label recall. This is the
same device as the "temporary model" class prior of Ben-Baruch et al.
([arXiv:2110.10955](https://arxiv.org/abs/2110.10955)). Against the survey's
seven human-audited family recalls it scores Spearman **+0.786** — the same
ballpark as the 0.81 that paper reports for its own estimator, which is
reassuring, because that paper turned an estimate of that quality into a real
gain.

It has a known bias, and the bias is visible: the level is compressed with a fit
slope of 1.78 and up to 27pp of error on the worst family. Two causes, both
identified: their temporary model is trained in Ignore mode and ours in Negative
mode, which they explicitly warn compresses frequent under-annotated classes; and
SCAR is violated. The SAR propensity correction of
[Bekker & Davis](https://arxiv.org/pdf/1809.03207) was tried and **does not
help** here — the model's score on labelled positives is flat across annotator
diligence (median `e_q5 - e_q1` = −0.021), because at low diligence the surviving
labelled positives are the blatant ones, and the two effects cancel.

**Diligence elasticity.** How steeply a tag's application rate rises with how
thoroughly the post was tagged, from the survey. Spearman **+0.929** against the
same human recalls, so it wins on general tags — but it is flat on character
tags (median `recall_proxy` 1.013), whose omission is an identification-latency
effect rather than a diligence effect.

**Combination.** Diligence for general, Elkan-Noto for character, ranked within
category, rating tags pinned to fully reliable. Spearman **+0.857** overall.

Qualitative check on where the resulting focusing parameter lands:

- hardest softened (gamma = 7): `frilled_skirt`, `frilled_sleeves`, `buckle`,
  `belt_buckle`, `sleeve_cuffs`, `collared_jacket`, `layered_dress`,
  `lace-trimmed_bra`, `arm_at_side`, `legs_together` — fine-grained garment and
  pose detail that only a meticulous tagger adds
- least softened (gamma = 2): `portrait`, `no_humans`, `robot`, `scenery`,
  `mecha`, `doll`, `waitress` — salient, defining, impossible to forget
- named probes: `1girl` 2.38, `solo` 2.43, `monochrome` 2.38 against `standing`
  5.85, `shoes` 5.75, `indoors` 5.25, `looking_at_viewer` 4.40 — exactly the
  family the human audit put at 25.4% recall

## 4 The losses

`animetimm/multilabel/losses.py`. Every loss returns an unreduced `(B, C)`
tensor and accepts soft targets, because mixup is on by default here.

```
L = sum_pos  L_F(p, gamma_pos)
  + sum_neg  keep * L_F(1 - p_shifted, gamma_neg_t)
L_F(p, g) = (1 - p)^g * (-log p)
```

- **`bce`** — the default. `BCEWithLogitsLoss(reduction='none')`, untouched.
- **`asl`** — Asymmetric Loss. Community standard for booru taggers
  (`gamma_neg=4, clip=0.05`).
- **`pasl`** — three switchable mechanisms:
  - *per-tag focusing*: `gamma_neg_t` interpolated between `gamma_neg` and
    `gamma_unann` by the reliability prior;
  - *per-sample top-k gate*: the k highest-scoring negatives of each sample are
    dropped. Positives are never gated;
  - *diligence modulation*: k is scaled up for samples carrying few labels. The
    sample's diligence is read off the label vector itself, so this needs no
    dataset change.

One deliberate divergence from CSL. Their class-prior gate drops a whole class's
un-annotated entries; here *every* non-positive is an assumed negative, so a
class-level hard ignore would mean never learning when the tag is absent. The
per-tag signal is therefore continuous, and hard dropping is reserved for the
per-sample gate.

Two entry points for the prior, explicit beating implicit: `--loss-prior-file`,
or a column of the dataset's own `tags.parquet`, which rides along in
`TagsInfo.df`. The second is the intended long-term home.

Numerical note: log-probabilities come from `logsigmoid`, not `log(sigmoid(.))`.
The latter loses the low tail to float32 rounding in exactly the regime a
confident negative occupies, and the unit tests catch it.

## 5 The experiment

`tools/loss_ablation.py`, `tools/run_ablation.sh`. Six arms, each adding one
mechanism, identical backbone / seed / data order / schedule / step budget:

| arm | what it adds |
|---|---|
| `bce` | baseline, current path |
| `asl` | `gamma_neg=4`, community standard |
| `asl_matched` | `gamma_neg=4.5`, the mean of the per-tag schedule |
| `pasl_gamma` | per-tag gamma from the reliability prior |
| `pasl_topk` | + per-sample top-16 negative gate |
| `pasl_dilig` | + gate scaled by sample diligence |

`asl_matched` exists so that any gain from `pasl_gamma` cannot be explained as
"a higher average gamma" rather than "gamma that varies with reliability".

AdamW makes the comparison scale-invariant: ASL's total loss is an order of
magnitude smaller than BCE's, which under SGD would silently change the effective
learning rate, but Adam's update `lr * m / (sqrt(v) + eps)` is invariant to a
constant rescaling of the loss and AdamW's weight decay is decoupled.

### Three evaluation harnesses, because the obvious one is compromised

- **audited AUC** — the model ranked against 1,367 human verdicts on 150 images
  (737 true, 630 false). The only metric where correctly predicting a tag
  danbooru forgot counts as correct. Threshold-free, so a calibration shift
  cannot game it.
- **diligent F1** — dbv4 labels restricted to test images above the 80th
  percentile of general-tag count. Content-biased, but common-mode across arms.
- **standard F1** — the whole test split. Reported because it is what the model
  cards show, and **expected to disagree**: a model that recovers missing tags is
  penalised here.

A method that lifts audited AUC while dropping standard F1 is doing what it was
designed to do. Reading standard F1 alone would reject it.

## 6 Results

All six arms completed at 5,000 steps, then were re-scored through one code path
(`tools/posthoc_eval.py`) with a 0.05–0.95 threshold sweep. Full write-up with
audit data lives in the survey repo as `losses.md`; the frozen per-pair scores
are `raw/covar/ablation_audited_scores.npz` there.

**Threshold-matched comparison is mandatory, and the two F1s need separate
thresholds.** ASL shifts the whole score distribution up, so the arms' optima sit
on *opposite sides* of the in-run 0.3/0.4/0.5 grid — micro peaks at 0.20 for BCE
and 0.70–0.75 for the ASL family, macro at 0.05 and 0.55–0.60. Three grid points,
three different rankings. Quoting one "best threshold" per arm is also wrong:
micro and macro peak 0.10–0.15 apart, and reading micro at macro's 0.60 shows
`asl_matched` at 0.416 when its real optimum is 0.515 at 0.725.

One artefact ruled out along the way: the in-run evaluation scores the *prepared*
module, so it runs under bf16 autocast, while `posthoc_eval.py` is fp32 on a bare
module. Reading the same weights on a 0.025 grid in both dtypes puts every
micro/macro difference within ±0.003, and the score mass in the disputed
0.50–0.55 band at 0.554% vs 0.558%. Precision is not a variable here.
`tools/threshold_curve_probe.py`.

**Two budgets, and the second one reverses a conclusion.** All six arms ran at
5,000 steps; `bce` / `asl_matched` / `pasl_gamma` were then re-run at 12,000. At
12,000 steps:

| arm | audited AUC | dbv4 micro-F1 | @th | dbv4 macro-F1 | @th | diligent macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| `bce` | 0.8599 | 0.5382 | 0.25 | 0.1510 | 0.05 | 0.2316 |
| `asl_matched` | 0.9099 | **0.5666** | 0.75 | **0.2975** | 0.65 | **0.3106** |
| `pasl_gamma` | **0.9162** | 0.5504 | 0.75 | 0.2714 | 0.65 | 0.3029 |

and at 5,000:

| arm | audited AUC | dbv4 micro-F1 | @th | dbv4 macro-F1 | @th | diligent macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| `bce` | 0.7435 | 0.4444 | 0.20 | 0.0215 | 0.05 | 0.0604 |
| `asl` | 0.8709 | 0.5119 | 0.70 | 0.1504 | 0.55 | 0.2051 |
| `asl_matched` | **0.8733** | 0.5113 | 0.70 | **0.1620** | 0.60 | **0.2123** |
| `pasl_gamma` | 0.8704 | 0.4871 | 0.70 | 0.1287 | 0.60 | 0.1945 |
| `pasl_topk` | 0.8410 | 0.4596 | 0.75 | 0.1305 | 0.60 | 0.1970 |
| `pasl_dilig` | 0.8306 | 0.4434 | 0.75 | 0.1305 | 0.60 | 0.1967 |

**The pooled audited AUC hides the effect this design was built for.** Of the
1,367 audited pairs, 1,257 (92.0%) are ones where the human agrees with dbv4 — a
model trained on dbv4 gets those right by construction. Only 110 pairs are in
conflict: 86 omissions and 24 false positives. Splitting them out, per-tag gamma
against the same-mean constant-gamma control:

| contrast (n) | 12,000 steps | 5,000 steps |
|---|---|---|
| omission recovery, 86 vs 606 (692) | **+0.0312 [+0.0138, +0.0501]** sig | +0.0242 [+0.0079, +0.0430] sig |
| false-positive suppression, 651 vs 24 (675) | +0.0158 [−0.0182, +0.0505] n.s. | +0.0195 [−0.0167, +0.0563] n.s. |
| pure disagreement, 86 vs 24 (110) | +0.0921 [+0.0353, +0.1566] sig | +0.0809 [+0.0215, +0.1430] sig |

**Per-tag gamma replicates and strengthens.** Both arms average gamma = 4.50
over the vocabulary, so this isolates *per-tag* gamma from *higher mean* gamma,
and doubling the budget grew the effect rather than washing it out. It is also
not buying recall with precision — its false-positive-contrast AUC is the
highest of any arm at both budgets (0.7499 / 0.7476).

**What the second budget kills: asymmetry is not noise-robustness.** On the same
omission contrast, `asl_matched` − `bce` was +0.0231 [+0.0017, +0.0466] and
significant at 5,000 steps; at 12,000 it is **−0.0011 [−0.0171, +0.0142], not
significant**, and on the pure-disagreement contrast it is **significantly worse
than BCE** (−0.0703 [−0.1275, −0.0155]). BCE simply needs longer to push
rare-tag logits over a threshold: its long-tail macro-F1 goes from exactly
0.0000 in the six lowest frequency deciles at 5,000 steps to 0.0658–0.1581 at
12,000 (8.6× on macro-F1 overall), while `asl_matched` gains only 1.4×. A
single-budget ablation reads that convergence gap as noise robustness.

**What survives: ASL's real win is convergence and calibration, and it is
large.** Its dbv4 macro-F1 advantage does not narrow with budget — the absolute
gap is 0.1405 at 5,000 steps and 0.1465 at 12,000 — and per frequency decile the
advantage is biggest in the tail (d1: 0.0658 → 0.2493, 3.8×; d10: 1.14×). That
is exactly the shape a higher negative focusing parameter should produce. Ship it
for that reason, not as a noise measure.

**Both hard gates are harmful.** Marginal effects on the omission contrast at
5,000 steps: top-k gate −0.0319 [−0.0494, −0.0152]; diligence modulation a
further −0.0111 [−0.0217, −0.0013]. Softening gamma helps; dropping cells does
not. The 16 negatives a model finds most suspicious in a picture are largely
*real* negatives, and deleting them removes the tag's only negative signal.
Neither gate was re-run at 12,000 steps, so these two are the weakest results
here.

**Verdict.** Ship `asl` as an opt-in with per-tag gamma available through the
prior column; do not ship either gate. Whichever of `asl_matched` /
`pasl_gamma` is right depends on the objective — reproduce danbooru's labels, or
label what is in the picture — and that choice has to be made explicitly,
because the default metric silently makes it for you. And do not select a loss
from a single step budget.

## 7 Limits

- One seed per arm within the time budget; the audited AUC has a standard error
  of roughly 0.02 on 1,367 pairs, so differences below about 0.04 are not
  readable. Bootstrap intervals are computed from per-pair scores in the
  post-hoc uniform re-evaluation.
- A short run on a small backbone. The cross-model analysis of §2 justifies the
  small backbone — per-tag behaviour is 97% a shared data property — but does not
  guarantee the loss ranking transfers to a 100-epoch run at full scale.
- The reliability prior is derived from a model trained in Negative mode, which
  the CSL authors warn compresses the estimate. Retraining the prior model in
  Ignore mode is the obvious next improvement and was not attempted here.
