"""Compare the loss-ablation arms across the three evaluation harnesses.

Read the harnesses in this order, because they disagree on purpose:

  audited AUC   the model ranked against 1,367 human verdicts.  The only metric
                where correctly predicting a tag danbooru forgot counts as
                right, so it is the only one that can reward a loss designed to
                stop treating missing labels as negatives.
  diligent F1   dbv4 labels, restricted to thoroughly-tagged test images.  Much
                less contaminated than the standard split, still free.
  standard F1   dbv4 labels on the whole test split.  Reported because it is
                what the model cards show, and expected to *disagree* with the
                other two: a model that recovers missing tags is penalised here.

A method that lifts audited AUC while dropping standard F1 is doing exactly what
it was built to do.  Reading standard F1 alone would reject it.
"""
import glob
import json
import os

import click


def load(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, '*.json'))):
        arm = os.path.basename(p)[:-5]
        if arm.endswith('_counts'):
            continue
        out[arm] = json.load(open(p))
    return out


ORDER = ['bce', 'asl', 'asl_matched', 'pasl_gamma', 'pasl_topk', 'pasl_dilig']
WHAT = {
    'bce': 'baseline, current animetimm path',
    'asl': 'ASL gamma_neg=4 (community standard)',
    'asl_matched': 'ASL gamma_neg=4.5 (matches per-tag mean; isolates level)',
    'pasl_gamma': '+ per-tag gamma from label reliability',
    'pasl_topk': '+ per-sample top-16 negative gate',
    'pasl_dilig': '+ gate scaled by how thoroughly the sample was tagged',
}


@click.command()
@click.option('--dir', 'd', default='/data/narugo1992/loss_ablation')
@click.option('--threshold', default=0.4, type=float)
def main(d, threshold):
    res = load(d)
    if not res:
        raise SystemExit(f'no results in {d}')
    arms = [a for a in ORDER if a in res] + [a for a in res if a not in ORDER]
    base = res.get('bce')

    th = f'th{threshold}'
    print(f'{"arm":13s}{"audited":>9s}{"bal.acc":>9s} | {"diligent":>9s}{"macro":>8s} | '
          f'{"standard":>9s}{"macro":>8s}  what')
    print(f'{"":13s}{"AUC":>9s}{"@" + str(threshold):>9s} | {"micro F1":>9s}{"F1":>8s} | '
          f'{"micro F1":>9s}{"F1":>8s}')
    print('-' * 104)
    for a in arms:
        r = res[a]
        au = r.get('audited', {})
        m = r['metrics']
        dil = m.get(f'diligent_{th}_all', {})
        std = m.get(f'standard_{th}_all', m.get(f'{th}_all', {}))
        print(f'{a:13s}{au.get("auc_vs_human", float("nan")):9.4f}'
              f'{au.get(th, {}).get("balanced_acc", float("nan")):9.4f} | '
              f'{dil.get("micro_f1", float("nan")):9.4f}'
              f'{dil.get("macro_f1", float("nan")):8.4f} | '
              f'{std.get("micro_f1", float("nan")):9.4f}'
              f'{std.get("macro_f1", float("nan")):8.4f}  {WHAT.get(a, "")}')

    if base:
        print('\ndeltas vs bce (positive = better):')
        b_au = base.get('audited', {}).get('auc_vs_human')
        b_dil = base['metrics'].get(f'diligent_{th}_all', {})
        b_std = base['metrics'].get(f'standard_{th}_all', {})
        for a in arms:
            if a == 'bce':
                continue
            r = res[a]
            au = r.get('audited', {}).get('auc_vs_human')
            dil = r['metrics'].get(f'diligent_{th}_all', {})
            std = r['metrics'].get(f'standard_{th}_all', {})
            print(f'  {a:13s} audited AUC {au - b_au:+.4f}   '
                  f'diligent micro {dil["micro_f1"] - b_dil["micro_f1"]:+.4f} '
                  f'macro {dil["macro_f1"] - b_dil["macro_f1"]:+.4f}   '
                  f'standard micro {std["micro_f1"] - b_std["micro_f1"]:+.4f} '
                  f'macro {std["macro_f1"] - b_std["macro_f1"]:+.4f}')

    print('\nsensitivity / specificity against human verdicts:')
    for a in arms:
        au = res[a].get('audited', {})
        row = '  ' + f'{a:13s}'
        for t in ('th0.3', 'th0.4', 'th0.5'):
            e = au.get(t)
            if e:
                row += f'  {t}: sens {e["sensitivity"]:.3f} spec {e["specificity"]:.3f}'
        print(row)

    one = res[arms[0]]
    print(f'\nsetup: {one["model"]}, {one["steps"]} steps, bs {one["batch_size"]}/gpu, '
          f'seed {one["seed"]}, {one["train_seconds"] / 60:.0f} min/arm, '
          f'eval on {one["eval_images"]} images '
          f'({one.get("diligent_images", "?")} of them thoroughly tagged), '
          f'{one.get("audited", {}).get("n_pairs", "?")} human-judged pairs.')


if __name__ == '__main__':
    main()
