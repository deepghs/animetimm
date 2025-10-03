import json
import os
from functools import partial

import click
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from ditk import logging
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
from tqdm import tqdm

from animetimm.utils import GLOBAL_CONTEXT_SETTINGS, print_version
from .dataset import load_tags, load_dataloader
from ..dataset import load_pretrained_tag
from ..model import Model


def test(workdir: str, num_workers: int = 32, batch_size: int = 32, force: bool = False, use_test_size: bool = True):
    if os.path.exists(os.path.join(workdir, 'test_tags.csv')) and not force:
        logging.info(f'Already tested for {workdir}, skipped.')
        return

    accelerator = Accelerator(
        # mixed_precision=self.cfgs.mixed_precision,
        step_scheduler_with_optimizer=False,
    )

    with open(os.path.join(workdir, 'meta.json'), 'r') as f:
        meta_info = json.load(f)

    dataset_repo_id = meta_info['train']['dataset']
    image_key = meta_info['train'].get('image_key') or 'webp'
    filters = dict(meta_info['train'].get('filters') or {})
    cof = meta_info['train'].get('cof', 1.0)
    tag_key = meta_info['train']['tag_key']

    tags_info = load_tags(repo_id=dataset_repo_id, filters=filters, cof=cof)
    if os.path.exists(os.path.join(workdir, 'tags.csv')):
        df_expected_tags = pd.read_csv(os.path.join(workdir, 'tags.csv'), keep_default_na=False)
        if len(tags_info.df) != len(df_expected_tags):
            raise RuntimeError('Tag list length not match, '
                               f'{len(df_expected_tags)!r} expected but {len(tags_info.df)!r} found.')
        elif list(tags_info.df['name']) != list(df_expected_tags['name']):
            raise RuntimeError('Tag list not match.')

    accelerator.wait_for_everyone()

    tags_info.df.to_csv(os.path.join(workdir, 'tags.csv'), index=False)
    checkpoints = os.path.join(workdir, 'checkpoints')
    best_ckpt_zip_file = os.path.join(checkpoints, 'best.zip')
    model, meta, metrics = Model.load_from_zip(best_ckpt_zip_file)
    previous_epoch = meta['step']
    if accelerator.is_main_process:
        logging.info(f'Resume from epoch {previous_epoch!r}.')

    model: Model
    pretrained_tag = meta_info['train'].get('pretrained_tag') or load_pretrained_tag(dataset_repo_id)
    logging.info(f'Pretrained tag {pretrained_tag!r} found for dataset {dataset_repo_id!r}.')
    model.pretrained_tag = pretrained_tag

    module = model.module
    test_dataloader = load_dataloader(
        repo_id=dataset_repo_id,
        model=module,
        split='test',
        batch_size=batch_size,
        num_workers=num_workers,
        pre_align=meta_info['train']['pre_align'],
        align_size=meta_info['train']['align_size'],
        is_main_process=accelerator.is_main_process,
        image_key=image_key,
        tag_key=tag_key,
        tag_filters=filters,
        use_test_size_when_test=use_test_size,
    )

    module, test_dataloader = accelerator.prepare(module, test_dataloader)
    if accelerator.is_main_process:
        logging.info(f'Model Class: {type(module)!r}')
        logging.info('Testing start!')

    module.eval()

    with torch.no_grad():
        eval_total = 0
        eval_top1, eval_top5 = 0, 0

        labs, preds = [], []
        for i, (inputs, labels_) in enumerate(tqdm(test_dataloader, disable=not accelerator.is_main_process)):
            inputs = inputs.float()
            labels_ = labels_

            outputs = module(inputs)
            eval_total += labels_.shape[0]

            as_ = torch.argsort(outputs, dim=-1)
            # print(as_.device, labels_.device, type(eval_top1))
            eval_top1 += (as_[:, -1] == labels_).sum().detach().cpu().item()
            as_top5 = as_[:, -5:]
            for t5, expected in zip(as_top5.detach().cpu().tolist(), labels_.detach().cpu().tolist()):
                if expected in t5:
                    eval_top5 += 1

            labs.append(labels_.clone().detach())
            preds.append(torch.argmax(outputs, dim=-1).detach())

        accelerator.wait_for_everyone()

        labs = torch.concat(labs)
        preds = torch.concat(preds)
        eval_total = accelerator.gather(
            torch.tensor([eval_total], device=accelerator.device)).sum().detach().cpu().item()

        labs = accelerator.gather(labs).detach().cpu().numpy()
        preds = accelerator.gather(preds).detach().cpu().numpy()

        eval_top1 = accelerator.gather(
            torch.tensor([eval_top1], device=accelerator.device)).sum().detach().cpu().item()
        eval_top5 = accelerator.gather(
            torch.tensor([eval_top5], device=accelerator.device)).sum().detach().cpu().item()

        if accelerator.is_main_process:
            macro_f1 = f1_score(labs, preds, average='macro', zero_division=0.0)
            macro_precision = precision_score(labs, preds, average='macro', zero_division=0.0)
            macro_recall = recall_score(labs, preds, average='macro', zero_division=0.0)

            micro_f1 = f1_score(labs, preds, average='micro', zero_division=0.0)
            micro_precision = precision_score(labs, preds, average='micro', zero_division=0.0)
            micro_recall = recall_score(labs, preds, average='micro', zero_division=0.0)

            l_precision, l_recall, l_f1, _ = precision_recall_fscore_support(
                labs,
                preds,
                labels=np.arange(0, len(tags_info.tags)),
                average=None,
                zero_division=0.0
            )

            df_tags_details = pd.DataFrame({
                **{name: tags_info.df[name] for name in tags_info.df.columns},
                'f1': l_f1,
                'precision': l_precision,
                'recall': l_recall,
            })
            _metrics = {
                'top-1': eval_top1 * 1.0 / eval_total,
                'top-5': eval_top5 * 1.0 / eval_total,
                'micro_f1': micro_f1,
                'micro_precision': micro_precision,
                'micro_recall': micro_recall,
                'macro_f1': macro_f1,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
            }

            logging.info(f'Metrics: {_metrics!r}')
            logging.info(f'Tag detailed information:\n{df_tags_details}')

            with open(os.path.join(workdir, 'test_metrics.json'), 'w') as f:
                json.dump(_metrics, f, sort_keys=True, ensure_ascii=False, indent=4)
            df_tags_details.to_csv(os.path.join(workdir, 'test_tags.csv'), index=False)


@click.command(context_settings={**GLOBAL_CONTEXT_SETTINGS}, help="Calculating test metrics for multilabel taggers.")
@click.option('-v', '--version', is_flag=True,
              callback=partial(print_version, 'animetimm.multilabel.test'), expose_value=False, is_eager=True)
@click.option('--num-workers', '-nw', default=32, type=int, help='Number of workers', show_default=True)
@click.option('--batch-size', '-bs', default=32, type=int, help='Batch size', show_default=True)
@click.option('--workdir', '-w', default=None, type=str, help='Workdir to save training data', show_default=True)
@click.option('--force/--non-force', default=True, help='Force re-calculate.', show_default=True)
@click.option('--use-test-size/--use-eval-size', 'use_test_size', default=True, help='Use test size for inference',
              show_default=True)
def cli(workdir, num_workers, batch_size, force, use_test_size):
    logging.try_init_root(logging.INFO)
    test(
        workdir=workdir,
        num_workers=num_workers,
        batch_size=batch_size,
        force=force,
        use_test_size=use_test_size,
    )


if __name__ == '__main__':
    cli()
