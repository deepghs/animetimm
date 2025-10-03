import json
import os
import warnings
from functools import partial
from typing import Optional, Sequence, List

import click
import pandas as pd
import torch
from accelerate import Accelerator
from ditk import logging
from tqdm import tqdm

from .dataset import load_tags, load_dataloader
from .metrics import mcc, f1score, precision, recall, compute_optimal_thresholds, \
    compute_optimal_thresholds_by_categories
from ..dataset import load_pretrained_tag
from ..model import Model
from ..utils import GLOBAL_CONTEXT_SETTINGS, print_version


def test(workdir: str, num_workers: int = 32, batch_size: int = 32, test_threshold: float = 0.4,
         tag_categories: Optional[Sequence[int]] = None, seen_tag_keys: Optional[List[str]] = None,
         force: bool = False, use_test_size: bool = True, accelerator: Optional[Accelerator] = None):
    if os.path.exists(os.path.join(workdir, 'test_tags.csv')) and not force:
        logging.info(f'Already tested for {workdir}, skipped.')
        return

    accelerator = accelerator or Accelerator(
        # mixed_precision=self.cfgs.mixed_precision,
        step_scheduler_with_optimizer=False,
    )

    with open(os.path.join(workdir, 'meta.json'), 'r') as f:
        meta_info = json.load(f)

    dataset_repo_id = meta_info['train']['dataset']
    image_key = meta_info['train'].get('image_key') or 'webp'
    tag_categories = tag_categories or meta_info['train'].get('tag_categories')
    seen_tag_keys = seen_tag_keys or meta_info['train'].get('seem_tag_keys') or meta_info['train'].get('seen_tag_keys')
    use_normalize = meta_info['train'].get('use_normalize', False) or False
    if accelerator.is_main_process:
        logging.info(f'Tags categories: {tag_categories!r}, seen tag keys: {seen_tag_keys!r}')

    tags_info = load_tags(repo_id=dataset_repo_id, categories=tag_categories)
    if os.path.exists(os.path.join(workdir, 'tags.csv')):
        df_expected_tags = pd.read_csv(os.path.join(workdir, 'tags.csv'), keep_default_na=False)
        if len(tags_info.df) != len(df_expected_tags):
            raise RuntimeError('Tag list length not match, '
                               f'{len(df_expected_tags)!r} expected but {len(tags_info.df)!r} found.')
        elif list(tags_info.df['name']) != list(df_expected_tags['name']):
            err_cnt = 0
            for i, (ls_tag, tag) in enumerate(zip(tags_info.df['name'], df_expected_tags['name'])):
                if ls_tag != tag:
                    warnings.warn(f'Tag list not match on #{i}, {ls_tag!r} vs {tag!r}.')
                    err_cnt += 1
                    if err_cnt >= 10:
                        raise RuntimeError('Too many tags not match.')

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
    if accelerator.is_main_process:
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
        categories=tag_categories,
        seen_tag_keys=seen_tag_keys,
        image_key=image_key,
        use_test_size_when_test=use_test_size,
        use_normalize=use_normalize,
    )

    module, test_dataloader = accelerator.prepare(module, test_dataloader)
    if accelerator.is_main_process:
        logging.info(f'Model Class: {type(module)!r}')
        logging.info('Testing start!')

    module.eval()

    with torch.no_grad():
        micro_tp = torch.zeros((1,), device=accelerator.device)
        micro_fp = torch.zeros((1,), device=accelerator.device)
        micro_tn = torch.zeros((1,), device=accelerator.device)
        micro_fn = torch.zeros((1,), device=accelerator.device)

        macro_tp = torch.zeros((1, len(tags_info.tags),), device=accelerator.device)
        macro_fp = torch.zeros((1, len(tags_info.tags),), device=accelerator.device)
        macro_tn = torch.zeros((1, len(tags_info.tags),), device=accelerator.device)
        macro_fn = torch.zeros((1, len(tags_info.tags),), device=accelerator.device)

        all_samples = []
        all_labels = []

        for i, (inputs, labels_) in enumerate(tqdm(test_dataloader, disable=not accelerator.is_main_process)):
            inputs = inputs.float()
            labels_ = labels_

            outputs = module(inputs)
            labels = labels_ > test_threshold
            preds = torch.sigmoid(outputs) > test_threshold
            micro_tp += ((preds == 1) & (labels == 1)).sum().item()
            micro_fp += ((preds == 1) & (labels == 0)).sum().item()
            micro_tn += ((preds == 0) & (labels == 0)).sum().item()
            micro_fn += ((preds == 0) & (labels == 1)).sum().item()
            macro_tp += ((preds == 1) & (labels == 1)).sum(dim=0)
            macro_fp += ((preds == 1) & (labels == 0)).sum(dim=0)
            macro_tn += ((preds == 0) & (labels == 0)).sum(dim=0)
            macro_fn += ((preds == 0) & (labels == 1)).sum(dim=0)

            all_samples.append(torch.sigmoid(outputs))
            all_labels.append(labels_)

            if i % 10 == 0:
                accelerator.wait_for_everyone()

        logging.info(f'Inference ready for #{accelerator.process_index}.')
        accelerator.wait_for_everyone()

        all_samples = torch.concat(all_samples, dim=0)
        all_labels = torch.concat(all_labels, dim=0)
        all_samples = accelerator.gather(all_samples).cpu()
        all_labels = accelerator.gather(all_labels).cpu()
        # all_samples, all_labels = accelerator.gather_for_metrics((all_samples, all_labels))
        if accelerator.is_main_process:
            logging.info(f'Gathered all_samples, shape: {all_samples.shape!r}, '
                         f'dtype: {all_samples.dtype!r}, device: {all_samples.device!r}')
            logging.info(f'Gathered all_labels, shape: {all_labels.shape!r}, '
                         f'dtype: {all_labels.dtype!r}, device: {all_labels.device!r}')

        # print((torch.isclose(all_labels, 1.0) | torch.isclose(all_labels, 0.0)).all())
        # quit()

        # micro_tp = micro_tp.sum(dim=0)
        # micro_fp = micro_fp.sum(dim=0)
        # micro_tn = micro_tn.sum(dim=0)
        # micro_fn = micro_fn.sum(dim=0)
        #
        # macro_tp = macro_tp.sum(dim=0)
        # macro_fp = macro_fp.sum(dim=0)
        # macro_tn = macro_tn.sum(dim=0)
        # macro_fn = macro_fn.sum(dim=0)

        micro_tp = accelerator.gather(micro_tp).sum(dim=0)
        micro_fp = accelerator.gather(micro_fp).sum(dim=0)
        micro_tn = accelerator.gather(micro_tn).sum(dim=0)
        micro_fn = accelerator.gather(micro_fn).sum(dim=0)

        macro_tp = accelerator.gather(macro_tp).sum(dim=0)
        macro_fp = accelerator.gather(macro_fp).sum(dim=0)
        macro_tn = accelerator.gather(macro_tn).sum(dim=0)
        macro_fn = accelerator.gather(macro_fn).sum(dim=0)

        if accelerator.is_main_process:
            best_thresholds, best_f1, best_precision, best_recall = \
                compute_optimal_thresholds(all_samples, all_labels, alpha=1.0, max_workers=32)

            c_best_thresholds, c_best_f1, c_best_precision, c_best_recall = \
                compute_optimal_thresholds_by_categories(all_samples, all_labels, tags_info.df, alpha=1.0,
                                                         max_workers=8)

            micro_mcc = mcc(micro_tp, micro_fp, micro_tn, micro_fn).detach().cpu().item()
            micro_f1 = f1score(micro_tp, micro_fp, micro_tn, micro_fn).detach().cpu().item()
            micro_precision = precision(micro_tp, micro_fp, micro_tn, micro_fn).detach().cpu().item()
            micro_recall = recall(micro_tp, micro_fp, micro_tn, micro_fn).detach().cpu().item()

            macro_mcc = mcc(macro_tp, macro_fp, macro_tn, macro_fn).detach().cpu().item()
            macro_f1 = f1score(macro_tp, macro_fp, macro_tn, macro_fn).detach().cpu().item()
            macro_precision = precision(macro_tp, macro_fp, macro_tn, macro_fn).detach().cpu().item()
            macro_recall = recall(macro_tp, macro_fp, macro_tn, macro_fn).detach().cpu().item()

            macro_mcc_lst = mcc(macro_tp, macro_fp, macro_tn, macro_fn, mean=False).detach().cpu().tolist()
            macro_f1_lst = f1score(macro_tp, macro_fp, macro_tn, macro_fn,
                                   mean=False).detach().cpu().tolist()
            macro_precision_lst = precision(macro_tp, macro_fp, macro_tn, macro_fn,
                                            mean=False).detach().cpu().tolist()
            macro_recall_lst = recall(macro_tp, macro_fp, macro_tn, macro_fn,
                                      mean=False).detach().cpu().tolist()

            df_tags_details = pd.DataFrame({
                **{name: tags_info.df[name] for name in tags_info.df.columns},
                'mcc': macro_mcc_lst,
                'f1': macro_f1_lst,
                'precision': macro_precision_lst,
                'recall': macro_recall_lst,
                'best_f1': best_f1,
                'best_threshold': best_thresholds,
                'best_precision': best_precision,
                'best_recall': best_recall,
            })
            _metrics = {
                'micro_mcc': micro_mcc,
                'micro_f1': micro_f1,
                'micro_precision': micro_precision,
                'micro_recall': micro_recall,
                'macro_mcc': macro_mcc,
                'macro_f1': macro_f1,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'categories': [
                    {
                        'category': cate,
                        'best_f1': c_best_f1[cate],
                        'best_precision': c_best_precision[cate],
                        'best_recall': c_best_recall[cate],
                        'best_threshold': c_best_thresholds[cate],
                    }
                    for cate in c_best_f1.keys()
                ]
            }
            logging.info(f'Metrics: {_metrics!r}')
            logging.info(f'Tag detailed information:\n{df_tags_details}')

            with open(os.path.join(workdir, 'test_metrics.json'), 'w') as f:
                json.dump(_metrics, f, sort_keys=True, ensure_ascii=False, indent=4)
            df_tags_details.to_csv(os.path.join(workdir, 'test_tags.csv'), index=False)

            with open(os.path.join(workdir, 'test_options.json'), 'w') as f:
                json.dump({
                    'use_test_size': use_test_size,
                    'test_threshold': test_threshold,
                }, f, ensure_ascii=False, sort_keys=True, indent=4)


@click.command(context_settings={**GLOBAL_CONTEXT_SETTINGS}, help="Calculating test metrics for multilabel taggers.")
@click.option('-v', '--version', is_flag=True,
              callback=partial(print_version, 'animetimm.multilabel.test'), expose_value=False, is_eager=True)
@click.option('--num-workers', '-nw', default=32, type=int, help='Number of workers', show_default=True)
@click.option('--batch-size', '-bs', default=32, type=int, help='Batch size', show_default=True)
@click.option('--test-threshold', '-tt', default=0.4, type=float, help='Test threshold', show_default=True)
@click.option('--tag-categories', '-tc', multiple=True, type=int, help='Tag categories (multiple)', show_default=True)
@click.option('--seen-tag-keys', '-stk', multiple=True, help='Seen tag keys (multiple)', show_default=True)
@click.option('--workdir', '-w', default=None, type=str, help='Workdir to save training data', show_default=True)
@click.option('--force/--non-force', default=True, help='Force re-calculate.', show_default=True)
@click.option('--use-test-size/--use-eval-size', 'use_test_size', default=True, help='Use test size for inference',
              show_default=True)
def cli(workdir, num_workers, batch_size, test_threshold, tag_categories, seen_tag_keys, force, use_test_size):
    logging.try_init_root(logging.INFO)
    tag_categories_seq = list(tag_categories) if tag_categories else None
    seen_tag_keys_list = list(seen_tag_keys) if seen_tag_keys else None
    test(
        workdir=workdir,
        num_workers=num_workers,
        batch_size=batch_size,
        test_threshold=test_threshold,
        tag_categories=tag_categories_seq,
        seen_tag_keys=seen_tag_keys_list,
        force=force,
        use_test_size=use_test_size,
    )


if __name__ == '__main__':
    cli()

# if __name__ == '__main__':
#     logging.try_init_root(logging.INFO)
#     W = os.environ['W']
#     CATES = list(map(int, os.environ['CATES'].split(','))) if os.environ.get('CATES') else None
#     ST = list(os.environ['ST'].split(',')) if os.environ.get('ST') else None
#     test(
#         workdir=W,
#         tag_categories=CATES,
#         seen_tag_keys=ST,
#     )
