"""Uniform post-hoc evaluation of every ablation checkpoint.

The in-run evaluation used a fixed threshold grid of 0.3/0.4/0.5, and that turned
out to be unusable: ASL shifts the whole score distribution upward, so BCE is
already past its optimum at 0.3 while ASL is still climbing at 0.5.  Comparing
two arms at one threshold measures the threshold, not the loss.

This pass fixes it without touching the training runs.  Every checkpoint is
re-scored through the same code on the same images with a 0.05..0.95 sweep, and
each arm is then read at *its own* best threshold, which is what animetimm does
in production anyway (`thresholds.csv`).  The audited scores are kept per pair so
the human-ground-truth comparison can be bootstrapped in pairs rather than
independently.

    accelerate launch --num_processes 8 --mixed_precision bf16 tools/posthoc_eval.py
"""
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != _HERE]
sys.path.insert(0, os.path.dirname(_HERE))

import click
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from PIL import Image
from timm.data import create_transform, resolve_data_config
from tqdm import tqdm

from animetimm.model import Model
from animetimm.multilabel.dataset import load_dataloader, load_tags

DATASET = 'animetimm/danbooru-wdtagger-v4-w640-ws-full'
# Every path below is overridable, because none of them can be checked in: the
# survey repo, the audited thumbnails and the scratch output directory all live
# outside this tree.  The fallbacks are the author's machine.
SURVEY = os.environ.get('DBSURVEY_ROOT',
                        '/data/narugo1992/danbooru-label-quality-survey')
ABL_OUT = os.environ.get('ABLATION_OUT', '/data/narugo1992/loss_ablation')
AUDIT_IMGS = os.environ.get('DBSURVEY_AUDIT_IMAGES',
                            '/data/narugo1992/tagquality/out/imgs_calib_640')
THS = np.round(np.arange(0.05, 0.96, 0.05), 2)


@torch.no_grad()
def sweep(module, loader, n_tags, device, general_mask, diligent_min, limit):
    n_th = len(THS)
    tp, fp, fn = (torch.zeros(n_th, n_tags, device=device) for _ in range(3))
    dtp, dfp, dfn = (torch.zeros(n_th, n_tags, device=device) for _ in range(3))
    ths = torch.tensor(THS, device=device, dtype=torch.float32)
    n_img = n_dil = 0
    for i, (x, y) in enumerate(tqdm(loader, desc='sweep', leave=False)):
        if limit and i >= limit:
            break
        p = torch.sigmoid(module(x.float())).float()
        yb = y > 0.5
        dil = (yb & general_mask).sum(1) >= diligent_min
        for ti in range(n_th):
            pb = p > ths[ti]
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


def best_f1_by_decile(tp, fp, fn, decile, support_min=5):
    """Best-threshold macro F1 within each frequency decile.

    The audited harness only reaches tags whose median frequency is 43x the
    vocabulary median -- its 106-tag core is entirely top-decile, and 94.6% of
    the vocabulary sits below its rarest tag.  So it cannot say whether a loss
    change helps the tail, which is exactly where false negatives dominate and
    where macro-F1 lives.  This split answers that, at the cost of scoring
    against contaminated labels; the comparison across arms within a decile is
    still meaningful because the contamination is identical for all of them.
    """
    out = {}
    per = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    # tp + fn is the tag's positive count and is invariant in the threshold, so
    # read one row.  Summing over the threshold axis would inflate support by
    # len(THS) and quietly disable the filter.
    sup = (tp + fn)[0]
    for d in range(int(decile.max()) + 1):
        m = (decile == d) & (sup >= support_min)
        if not m.any():
            continue
        macro = per[:, m].mean(1)
        i = int(macro.argmax())
        out[f'd{d + 1}'] = dict(macro_f1=round(float(macro[i]), 5),
                                th=float(THS[i]), n_tags=int(m.sum()))
    return out


def best_f1(tp, fp, fn, support_min=1):
    """Micro and macro F1 at the sweep's best threshold for each."""
    micro = (2 * tp.sum(1) / (2 * tp.sum(1) + fp.sum(1) + fn.sum(1)).clamp(min=1))
    per = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    keep = (tp + fn)[0] >= support_min   # see best_f1_by_decile on why not .sum(0)
    macro = per[:, keep].mean(1) if keep.any() else torch.zeros(len(THS))
    i_mi, i_ma = int(micro.argmax()), int(macro.argmax())
    return dict(micro_f1=round(float(micro[i_mi]), 5), micro_th=float(THS[i_mi]),
                macro_f1=round(float(macro[i_ma]), 5), macro_th=float(THS[i_ma]),
                n_tags=int(keep.sum()))


@torch.no_grad()
def audited_scores(module, device, gt, img_dir, tags_to_id):
    trans = create_transform(**resolve_data_config({}, model=module),
                             is_training=False)
    imgs = sorted(gt['img'].unique())
    per = {}
    for i0 in range(0, len(imgs), 32):
        chunk = imgs[i0:i0 + 32]
        batch = torch.stack([trans(Image.open(os.path.join(img_dir, f'{i}.jpg'))
                                   .convert('RGB')) for i in chunk]).to(device)
        p = torch.sigmoid(module(batch.float())).float().cpu().numpy()
        for k, i in enumerate(chunk):
            per[i] = p[k]
    return np.array([per[r.img][tags_to_id[r.tag]] for r in gt.itertuples()])


def auc(y, s):
    df = pd.DataFrame({'s': s})
    ranks = df['s'].rank(method='average').to_numpy()
    n1, n0 = float(y.sum()), float((1 - y).sum())
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


@click.command()
@click.option('--dir', 'd', default=ABL_OUT)
@click.option('--model-name', default='caformer_s18.sail_in22k_ft_in1k_384')
@click.option('--batch-size', default=64, type=int)
@click.option('--num-workers', default=22, type=int)
@click.option('--align-size', default=448, type=int)
@click.option('--eval-batches', default=300, type=int)
@click.option('--diligent-min', default=40, type=int)
@click.option('--gt', 'gt_path', default=f'{SURVEY}/derived/covariates/audited_gt.parquet')
@click.option('--img-dir', default=AUDIT_IMGS)
def main(d, model_name, batch_size, num_workers, align_size, eval_batches,
         diligent_min, gt_path, img_dir):
    acc = Accelerator()
    tags_info = load_tags(DATASET)
    n_tags = len(tags_info.tags)
    gt = pd.read_parquet(gt_path)
    gt = gt[gt['tag'].isin(tags_info.tags_to_id)].reset_index(drop=True)

    proto = Model.new(model_name=model_name, tags=tags_info.tags, pretrained=False)
    loader = load_dataloader(DATASET, model=proto.module, split='test',
                             batch_size=batch_size, num_workers=num_workers,
                             pre_align=True, align_size=align_size,
                             is_main_process=acc.is_main_process)
    loader = acc.prepare(loader)
    general_mask = torch.tensor((tags_info.df['category'] == 0).to_numpy(),
                                device=acc.device)
    dec = pd.qcut(np.log10(tags_info.df['selected_count'].clip(lower=1)), 10,
                  labels=False, duplicates='drop').to_numpy()
    decile = torch.tensor(dec, device=acc.device)

    out = {}
    scores = {}
    for ckpt in sorted(glob.glob(os.path.join(d, '*.pt'))):
        arm = os.path.basename(ckpt)[:-3]
        model = Model.new(model_name=model_name, tags=tags_info.tags,
                          pretrained=False)
        model.module.load_state_dict(
            torch.load(ckpt, map_location='cpu', weights_only=True))
        module = model.module.to(acc.device).eval()
        std, dil, n_img, n_dil = sweep(module, loader, n_tags, acc.device,
                                       general_mask, diligent_min, eval_batches)
        std = [acc.reduce(t, reduction='sum') for t in std]
        dil = [acc.reduce(t, reduction='sum') for t in dil]
        if acc.is_main_process:
            out[arm] = {
                'standard_by_decile': best_f1_by_decile(*std, decile=decile),
                'diligent_by_decile': best_f1_by_decile(*dil, decile=decile),
                'standard': best_f1(*std),
                'standard_sup20': best_f1(*std, support_min=20),
                'diligent': best_f1(*dil),
                'diligent_sup20': best_f1(*dil, support_min=20),
            }
            scores[arm] = audited_scores(module, acc.device, gt, img_dir,
                                         tags_info.tags_to_id)
            print(f'  {arm}: swept')
        del model, module
        torch.cuda.empty_cache()
        acc.wait_for_everyone()

    if not acc.is_main_process:
        return
    y = gt['truth'].to_numpy()
    np.savez(os.path.join(d, 'audited_scores.npz'), y=y, **scores)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(y), size=(4000, len(y)))
    base = scores.get('bce')
    for arm in out:
        s = scores[arm]
        out[arm]['audited_auc'] = round(float(auc(y, s)), 5)
        if base is not None and arm != 'bce':
            db = np.array([auc(y[i], s[i]) - auc(y[i], base[i]) for i in idx])
            out[arm]['audited_dauc_vs_bce'] = round(float(np.mean(db)), 5)
            out[arm]['audited_dauc_ci95'] = [round(float(v), 5) for v in
                                             np.percentile(db, [2.5, 97.5])]
            out[arm]['audited_p_better'] = round(float((db > 0).mean()), 4)
    with open(os.path.join(d, 'posthoc.json'), 'w') as f:
        json.dump({'thresholds': THS.tolist(), 'eval_images': n_img,
                   'diligent_images': n_dil, 'arms': out}, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
