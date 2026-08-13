"""Controlled ablation of the training losses in `animetimm.multilabel.losses`.

The point is to isolate the loss.  Every arm sees the same backbone, the same
images in the same order, the same augmentation, the same schedule and the same
seed; only `loss_fn` differs.  That is why this lives outside `train.py` -- the
production path streams the full 5.3M-image split and runs for days, which
cannot answer a loss question inside a two-day budget.

Three evaluation harnesses, because the obvious one is compromised:

  standard      macro/micro F1 against dbv4's own test labels.  This is the
                number the model cards report, and it is contaminated: the test
                labels carry the same systematic omissions as the training
                labels, so a model that correctly predicts a missing tag is
                scored as wrong.
  diligent      the same metrics restricted to the top quintile of test images
                by label count.  Those images were tagged thoroughly, so far
                fewer of their negatives are false.  Content-biased, but the
                bias is common-mode across arms, which is all an ablation needs.
  audited       the 150 human-verified images of the danbooru label-quality
                survey, scored against the human verdicts rather than against
                dbv4.  Small (1,500 judgements) but the only true ground truth.

    python tools/loss_ablation.py --arm bce --steps 4000
"""
import json
import os
import sys
import time
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
# `tools/` contains a `timm/` package of its own, and python puts the script's
# directory first on sys.path -- which would shadow the real timm.  Drop it and
# add the repo root instead.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != _HERE]
sys.path.insert(0, os.path.dirname(_HERE))

import click
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from ditk import logging
from torch.optim import lr_scheduler
from tqdm import tqdm

from animetimm.model import Model
from animetimm.multilabel.dataset import load_dataloader, load_tags
from animetimm.multilabel.losses import build_loss, neg_gamma_from_scores

DATASET = 'animetimm/danbooru-wdtagger-v4-w640-ws-full'
# Every path below is overridable, because none of them can be checked in: the
# survey repo, the audited thumbnails and the scratch output directory all live
# outside this tree.  The fallbacks are the author's machine.
SURVEY = os.environ.get('DBSURVEY_ROOT',
                        '/data/narugo1992/danbooru-label-quality-survey')
ABL_OUT = os.environ.get('ABLATION_OUT', '/data/narugo1992/loss_ablation')
AUDIT_IMGS = os.environ.get('DBSURVEY_AUDIT_IMAGES',
                            '/data/narugo1992/tagquality/out/imgs_calib_640')

# The arms are nested so each one adds exactly one mechanism.  `asl_matched`
# exists to keep the comparison honest: the per-tag schedule averages gamma=4.50
# over the vocabulary, so without a constant-4.50 control any gain from
# `pasl_gamma` could just be "a higher average gamma" rather than "gamma that
# varies with how trustworthy the tag's absence is".
ARMS = {
    'bce': ('bce', {}),
    'asl': ('asl', dict(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)),
    'asl_matched': ('asl', dict(gamma_neg=4.5, gamma_pos=0.0, clip=0.05)),
    'pasl_gamma': ('pasl', dict(gamma_neg=2.0, gamma_pos=0.0, gamma_unann=7.0,
                                clip=0.05, use_prior=True)),
    'pasl_topk': ('pasl', dict(gamma_neg=2.0, gamma_pos=0.0, gamma_unann=7.0,
                               clip=0.05, use_prior=True, ignore_topk=16)),
    'pasl_dilig': ('pasl', dict(gamma_neg=2.0, gamma_pos=0.0, gamma_unann=7.0,
                                clip=0.05, use_prior=True, ignore_topk=16,
                                diligence_modulate=True)),
}


def build_arm(arm: str, tags, prior_file: Optional[str], gamma_neg_override=None):
    name, kw = ARMS[arm]
    kw = dict(kw)
    if kw.pop('use_prior', False):
        df = pd.read_parquet(prior_file)
        mapping = dict(zip(df['name'], df['reliability']))
        scores = np.array([mapping.get(t, np.nan) for t in tags], dtype=np.float32)
        kw['neg_gamma'] = neg_gamma_from_scores(
            scores, gamma_neg=kw['gamma_neg'], gamma_unann=kw['gamma_unann'])
    if gamma_neg_override is not None:
        kw['gamma_neg'] = gamma_neg_override
    return build_loss(name, **kw)


@torch.no_grad()
def evaluate(module, loader, n_tags, device, thresholds, general_mask,
             diligent_min=40, limit_batches=None):
    """Per-tag tp/fp/fn at each threshold, for all test images and for the
    thoroughly-tagged subset.

    ``diligent_min`` is the 80th percentile of general-tag count on the dbv4
    test split.  Images above it were tagged exhaustively enough that far fewer
    of their negatives are actually missing labels, so the same metric computed
    there is much less contaminated.  The subset is content-biased -- busy
    pictures attract diligent taggers -- but the bias is identical for every arm.
    """
    shape = (len(thresholds), n_tags)
    tp, fp, fn = (torch.zeros(shape, device=device) for _ in range(3))
    dtp, dfp, dfn = (torch.zeros(shape, device=device) for _ in range(3))
    n_img = n_dil = 0
    module.eval()
    for i, (x, y) in enumerate(tqdm(loader, desc='eval', leave=False)):
        if limit_batches and i >= limit_batches:
            break
        p = torch.sigmoid(module(x.float()))
        yb = y > 0.5
        dil = (yb & general_mask).sum(1) >= diligent_min
        for ti, th in enumerate(thresholds):
            pb = p > th
            tp[ti] += (pb & yb).sum(0)
            fp[ti] += (pb & ~yb).sum(0)
            fn[ti] += (~pb & yb).sum(0)
            if dil.any():
                dtp[ti] += (pb[dil] & yb[dil]).sum(0)
                dfp[ti] += (pb[dil] & ~yb[dil]).sum(0)
                dfn[ti] += (~pb[dil] & yb[dil]).sum(0)
        n_img += x.shape[0]
        n_dil += int(dil.sum())
    return (tp, fp, fn), (dtp, dfp, dfn), n_img, n_dil


@torch.no_grad()
def evaluate_audited(module, model, device, audited_gt, img_dir, tags_to_id,
                     thresholds):
    """Score the model against *human* verdicts, not against dbv4.

    1,367 (image, tag) pairs on 150 images were adjudicated by eye in the
    danbooru label-quality survey: 737 true, 630 false.  Everything else in this
    harness compares the model to labels that share the training set's
    omissions; this is the only place a correct prediction of a missing tag is
    scored as correct.  ROC-AUC is the headline because it needs no threshold
    and so cannot be gamed by a shift in calibration.
    """
    from PIL import Image
    from timm.data import resolve_data_config, create_transform
    cfg = resolve_data_config({}, model=model.module)
    trans = create_transform(**cfg, is_training=False)

    imgs = sorted(audited_gt['img'].unique())
    scores = {}
    for i0 in range(0, len(imgs), 32):
        chunk = imgs[i0:i0 + 32]
        batch = torch.stack([trans(Image.open(os.path.join(img_dir, f'{i}.jpg'))
                                   .convert('RGB')) for i in chunk]).to(device)
        p = torch.sigmoid(module(batch.float())).float().cpu().numpy()
        for k, i in enumerate(chunk):
            scores[i] = p[k]

    y, s = [], []
    for r in audited_gt.itertuples():
        j = tags_to_id.get(r.tag)
        if j is None:
            continue
        y.append(r.truth)
        s.append(float(scores[r.img][j]))
    y, s = np.array(y), np.array(s)

    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)

    out = {'n_pairs': int(len(y)), 'n_true': n1, 'n_false': n0,
           'auc_vs_human': round(float(auc), 5)}
    for th in thresholds:
        pred = s > th
        out[f'th{th}'] = {
            'sensitivity': round(float(pred[y == 1].mean()), 5),
            'specificity': round(float((~pred[y == 0]).mean()), 5),
            'balanced_acc': round(float(
                (pred[y == 1].mean() + (~pred[y == 0]).mean()) / 2), 5)}
    return out


def f1_from_counts(tp, fp, fn, support_min=0):
    micro = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()).clamp(min=1)
    sup = tp + fn
    keep = sup >= max(support_min, 1)
    per = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    macro = per[keep].mean() if keep.any() else torch.tensor(0.0)
    return float(micro), float(macro), int(keep.sum())


@click.command()
@click.option('--arm', type=click.Choice(list(ARMS)), required=True)
@click.option('--model-name', default='caformer_s18.sail_in22k_ft_in1k_384')
@click.option('--steps', default=4000, type=int, help='Optimizer steps (all arms equal).')
@click.option('--batch-size', default=48, type=int)
@click.option('--num-workers', default=20, type=int)
@click.option('--lr', default=2e-4, type=float)
@click.option('--seed', default=0, type=int)
@click.option('--align-size', default=448, type=int)
@click.option('--eval-batches', default=120, type=int)
@click.option('--prior-file', default=f'{SURVEY}/derived/covariates/tag_reliability.parquet')
@click.option('--diligent-min', default=40, type=int,
              help='General-tag count above which a test image counts as '
                   'thoroughly tagged (p80 of the dbv4 test split).')
@click.option('--audited-gt', default=f'{SURVEY}/derived/covariates/audited_gt.parquet')
@click.option('--audited-dir', default=AUDIT_IMGS)
@click.option('--out', default=ABL_OUT)
def main(arm, model_name, steps, batch_size, num_workers, lr, seed, align_size,
         eval_batches, prior_file, diligent_min, audited_gt, audited_dir, out):
    logging.try_init_root(logging.INFO)
    acc = Accelerator(step_scheduler_with_optimizer=False)
    torch.manual_seed(seed)
    np.random.seed(seed)

    tags_info = load_tags(DATASET)
    n_tags = len(tags_info.tags)
    model = Model.new(model_name=model_name, tags=tags_info.tags, pretrained=True,
                      model_args={'drop_path_rate': 0.1})
    module = model.module

    train_loader = load_dataloader(
        DATASET, model=module, split='train', batch_size=batch_size,
        num_workers=num_workers, noise_level=2, rotation_ratio=0.0,
        mixup_alpha=0.6, cutout_max_pct=0.0, cutout_patches=0,
        random_resize_method=True, pre_align=True, align_size=align_size,
        is_main_process=acc.is_main_process)
    test_loader = load_dataloader(
        DATASET, model=module, split='test', batch_size=batch_size,
        num_workers=num_workers, pre_align=True, align_size=align_size,
        is_main_process=acc.is_main_process)

    loss_fn = build_arm(arm, tags_info.tags, prior_file)
    if acc.is_main_process:
        logging.info(f'ARM {arm}: {loss_fn!r}')
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, module.parameters()),
                            lr=lr, weight_decay=1e-3)
    module, opt, train_loader, test_loader, loss_fn = acc.prepare(
        module, opt, train_loader, test_loader, loss_fn)
    sched = lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                    pct_start=0.15, final_div_factor=20.)
    label_weights = torch.from_numpy(tags_info.weights).to(acc.device)

    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    done, running = 0, 0.0
    module.train()
    pbar = tqdm(total=steps, disable=not acc.is_local_main_process, desc=f'train:{arm}')
    while done < steps:
        for x, y in train_loader:
            if done >= steps:
                break
            opt.zero_grad()
            outputs = module(x.float())
            loss = (loss_fn(outputs, y) * label_weights).sum()
            acc.backward(loss)
            opt.step()
            sched.step()
            running += loss.item()
            done += 1
            pbar.update(1)
            if done % 200 == 0 and acc.is_local_main_process:
                pbar.set_postfix(loss=f'{running / 200:.1f}', lr=f'{sched.get_last_lr()[0]:.2e}')
                running = 0.0
    pbar.close()
    train_secs = time.time() - t0

    ths = [0.3, 0.4, 0.5]
    general_mask = torch.tensor(
        (tags_info.df['category'] == 0).to_numpy(), device=acc.device)
    std, dil, n_img, n_dil = evaluate(
        module, test_loader, n_tags, acc.device, ths, general_mask,
        diligent_min=diligent_min, limit_batches=eval_batches)
    std = [acc.reduce(t, reduction='sum') for t in std]
    dil = [acc.reduce(t, reduction='sum') for t in dil]
    n_dil = int(acc.reduce(torch.tensor([n_dil], device=acc.device),
                           reduction='sum').item())
    tp, fp, fn = std

    if acc.is_main_process:
        res = {'arm': arm, 'steps': steps, 'model': model_name, 'seed': seed,
               'batch_size': batch_size, 'train_seconds': round(train_secs, 1),
               'eval_images': int(n_img) * acc.num_processes, 'metrics': {}}
        res['diligent_images'] = n_dil
        for name, cnt in (('standard', std), ('diligent', dil)):
            for ti, th in enumerate(ths):
                for smin, lab in ((1, 'all'), (20, 'sup20')):
                    mi, ma, nk = f1_from_counts(cnt[0][ti], cnt[1][ti], cnt[2][ti], smin)
                    res['metrics'][f'{name}_th{th}_{lab}'] = {
                        'micro_f1': round(mi, 5), 'macro_f1': round(ma, 5),
                        'n_tags': nk}
        gt = pd.read_parquet(audited_gt)
        res['audited'] = evaluate_audited(
            acc.unwrap_model(module), model, acc.device, gt, audited_dir,
            tags_info.tags_to_id, ths)
        logging.info(f'{arm} vs HUMAN ground truth: '
                     f'AUC={res["audited"]["auc_vs_human"]:.4f}')
        path = os.path.join(out, f'{arm}.json')
        with open(path, 'w') as f:
            json.dump(res, f, indent=1)
        np.savez(os.path.join(out, f'{arm}_counts.npz'),
                 tp=std[0].cpu().numpy(), fp=std[1].cpu().numpy(),
                 fn=std[2].cpu().numpy(), dtp=dil[0].cpu().numpy(),
                 dfp=dil[1].cpu().numpy(), dfn=dil[2].cpu().numpy(),
                 thresholds=np.array(ths))
        acc.unwrap_model(module).eval()
        torch.save(acc.unwrap_model(module).state_dict(),
                   os.path.join(out, f'{arm}.pt'))
        logging.info(f'{arm}: {json.dumps(res["metrics"], indent=1)}')
        logging.info(f'-> {path}')


if __name__ == '__main__':
    main()
