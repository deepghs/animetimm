"""Is the micro-F1 jump between threshold 0.50 and 0.55 real, or bf16?

The in-run evaluation in `loss_ablation.py` scores the model *after*
`Accelerator.prepare`, so with `--mixed_precision bf16` its forward pass runs
under autocast.  `posthoc_eval.py` loads the same weights into a bare module and
runs fp32.  For `asl_matched` at 5,000 steps the two disagree wildly at one
threshold -- in-run micro-F1 0.2443 at 0.50, post-hoc best 0.5113 at 0.55-0.60 --
and there are two very different explanations:

  real       ASL compresses the score distribution, so a large mass of cells sits
             in a narrow band and 0.05 of threshold genuinely moves them.
  artefact   bf16 has ~3 decimal digits of mantissa.  If that mass sits *at* the
             threshold, rounding alone decides which side it lands on, and the
             in-run number is measuring numeric noise.

Distinguishing them needs the same weights read on a fine grid in both dtypes.
Single GPU, small batch count -- this shares the box with a training run.

    python tools/threshold_curve_probe.py --ckpt /path/asl_matched.pt
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != _HERE]
sys.path.insert(0, os.path.dirname(_HERE))

import click
import numpy as np
import torch
from tqdm import tqdm

from animetimm.model import Model
from animetimm.multilabel.dataset import load_dataloader, load_tags

DATASET = 'animetimm/danbooru-wdtagger-v4-w640-ws-full'
THS = np.round(np.arange(0.30, 0.761, 0.025), 4)


@torch.no_grad()
def curve(module, loader, n_tags, device, batches, autocast_dtype=None):
    n_th = len(THS)
    tp, fp, fn = (torch.zeros(n_th, n_tags, device=device, dtype=torch.float64)
                  for _ in range(3))
    ths = torch.tensor(THS, device=device, dtype=torch.float32)
    band = torch.zeros(4, device=device, dtype=torch.float64)
    for i, (x, y) in enumerate(tqdm(loader, total=batches, leave=False)):
        if i >= batches:
            break
        x = x.to(device).float()
        if autocast_dtype is None:
            p = torch.sigmoid(module(x)).float()
        else:
            with torch.autocast('cuda', dtype=autocast_dtype):
                p = torch.sigmoid(module(x))
            p = p.float()
        yb = (y.to(device) > 0.5)
        for ti in range(n_th):
            pb = p > ths[ti]
            tp[ti] += (pb & yb).sum(0)
            fp[ti] += (pb & ~yb).sum(0)
            fn[ti] += (~pb & yb).sum(0)
        # how much probability mass sits in the disputed band
        band[0] += float((p < 0.45).sum())
        band[1] += float(((p >= 0.45) & (p < 0.50)).sum())
        band[2] += float(((p >= 0.50) & (p < 0.55)).sum())
        band[3] += float((p >= 0.55).sum())
    return tp, fp, fn, band


def micro_macro(tp, fp, fn):
    mi = 2 * tp.sum(1) / (2 * tp.sum(1) + fp.sum(1) + fn.sum(1)).clamp(min=1)
    per = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    keep = (tp + fn)[0] >= 1
    return mi.cpu().numpy(), per[:, keep].mean(1).cpu().numpy()


@click.command()
@click.option('--ckpt', required=True)
@click.option('--model-name', default='caformer_s18.sail_in22k_ft_in1k_384')
@click.option('--batch-size', default=32, type=int)
@click.option('--batches', default=40, type=int)
@click.option('--num-workers', default=6, type=int)
@click.option('--align-size', default=448, type=int)
@click.option('--device', default='cuda:7')
@click.option('--out', 'out_path', default=None, help='Write the curves to this json.')
def main(ckpt, model_name, batch_size, batches, num_workers, align_size, device,
         out_path):
    tags_info = load_tags(DATASET)
    n_tags = len(tags_info.tags)
    model = Model.new(model_name=model_name, tags=tags_info.tags, pretrained=False)
    model.module.load_state_dict(
        torch.load(ckpt, map_location='cpu', weights_only=True))
    module = model.module.to(device).eval()

    out = {}
    for lab, dt in (('fp32', None), ('bf16', torch.bfloat16)):
        loader = load_dataloader(DATASET, model=module, split='test',
                                 batch_size=batch_size, num_workers=num_workers,
                                 pre_align=True, align_size=align_size,
                                 is_main_process=True)
        tp, fp, fn, band = curve(module, loader, n_tags, device, batches, dt)
        out[lab] = (*micro_macro(tp, fp, fn), band.cpu().numpy())

    tot = out['fp32'][2].sum()
    print(f'\n{batches * batch_size} images, {n_tags} tags\n')
    print('score mass by band (fp32 / bf16, % of all cells)')
    for k, lab in enumerate(('<0.45', '0.45-0.50', '0.50-0.55', '>=0.55')):
        print(f'  {lab:>10}  {100 * out["fp32"][2][k] / tot:7.3f}  '
              f'{100 * out["bf16"][2][k] / tot:7.3f}')
    print(f'\n{"th":>6}{"fp32 micro":>12}{"bf16 micro":>12}{"delta":>9}'
          f'{"fp32 macro":>12}{"bf16 macro":>12}')
    for i, th in enumerate(THS):
        a, b = out['fp32'][0][i], out['bf16'][0][i]
        print(f'{th:6.3f}{a:12.4f}{b:12.4f}{b - a:+9.4f}'
              f'{out["fp32"][1][i]:12.4f}{out["bf16"][1][i]:12.4f}')
    i32, i16 = int(np.argmax(out['fp32'][0])), int(np.argmax(out['bf16'][0]))
    print(f'\nbest micro: fp32 {out["fp32"][0][i32]:.4f} @ {THS[i32]}, '
          f'bf16 {out["bf16"][0][i16]:.4f} @ {THS[i16]}')
    if out_path:
        import json
        payload = {
            'ckpt': os.path.basename(ckpt), 'model': model_name,
            'images': batches * batch_size, 'n_tags': n_tags,
            'thresholds': [float(t) for t in THS],
            'bands': ['<0.45', '0.45-0.50', '0.50-0.55', '>=0.55'],
            'dtypes': {lab: {
                'micro_f1': [round(float(v), 5) for v in out[lab][0]],
                'macro_f1': [round(float(v), 5) for v in out[lab][1]],
                'band_pct': [round(100 * float(v) / tot, 4) for v in out[lab][2]],
            } for lab in ('fp32', 'bf16')},
            'max_abs_dtype_delta_micro': round(float(np.max(np.abs(
                np.asarray(out['fp32'][0]) - np.asarray(out['bf16'][0])))), 5),
            'max_abs_dtype_delta_macro': round(float(np.max(np.abs(
                np.asarray(out['fp32'][1]) - np.asarray(out['bf16'][1])))), 5),
        }
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=1)
        print(f'-> {out_path}')


if __name__ == '__main__':
    main()
