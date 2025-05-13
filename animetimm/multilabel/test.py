import json
import os
from tempfile import TemporaryDirectory
from typing import Optional, Sequence, List

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from ditk import logging
from tqdm import tqdm

from .dataset import load_tags, load_pretrained_tag, load_dataloader
from .metrics import mcc, f1score, precision, recall, compute_optimal_thresholds, \
    compute_optimal_thresholds_by_categories
from ..model import Model


def test(workdir: str, num_workers: int = 32, batch_size: int = 32, test_threshold: float = 0.4,
         tag_categories: Optional[Sequence[int]] = None,
         seen_tag_keys: Optional[List[str]] = None, force: bool = False):
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
    tag_categories = tag_categories or meta_info['train'].get('tag_categories')

    tags_info = load_tags(repo_id=dataset_repo_id, categories=tag_categories)
    if os.path.exists(os.path.join(workdir, 'tags.csv')):
        df_expected_tags = pd.read_csv(os.path.join(workdir, 'tags.csv'))
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
        categories=tag_categories,
        seen_tag_keys=seen_tag_keys,
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

        logging.info(f'Inference ready for #{accelerator.process_index}.')
        accelerator.wait_for_everyone()

        all_samples = torch.concat(all_samples, dim=0).cpu()
        all_labels = torch.concat(all_labels, dim=0).cpu()
        with TemporaryDirectory() as td:
            blist = [td]
            broadcast_object_list(blist, from_process=0)
            td = blist[0]
            logging.info(f'Using temporary directory {td!r} for metrics syncing.')

            dst_pt_file = os.path.join(td, f'shard_{accelerator.process_index}.pt')
            logging.info(f'Saving data of #{accelerator.process_index} to {dst_pt_file!r} ...')
            torch.save({
                'samples': all_samples,
                'labels': all_labels,
            }, dst_pt_file)

            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                logging.info('Gathering metrics data to rank0 ...')
                all_samples, all_labels = [], []
                for i in tqdm(range(accelerator.num_processes)):
                    sh_info = torch.load(os.path.join(td, f'shard_{i}.pt'))
                    all_samples.append(sh_info['samples'])
                    all_labels.append(sh_info['labels'])
                all_samples = torch.concat(all_samples)
                all_labels = torch.concat(all_labels)

        # all_samples, all_labels = accelerator.gather_for_metrics((all_samples, all_labels))
        print('all_samples', all_samples.shape, all_samples.dtype, all_samples.device)
        print('all_labels', all_labels.shape, all_labels.dtype, all_labels.device)

        # print((torch.isclose(all_labels, 1.0) | torch.isclose(all_labels, 0.0)).all())
        # quit()

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
                compute_optimal_thresholds(all_samples, all_labels, alpha=1.0)

            c_best_thresholds, c_best_f1, c_best_precision, c_best_recall = \
                compute_optimal_thresholds_by_categories(all_samples, all_labels, tags_info.df, alpha=1.0)

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


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    W = os.environ['W']
    CATES = list(map(int, os.environ['CATES'].split(','))) if os.environ.get('CATES') else None
    ST = list(os.environ['ST'].split(',')) if os.environ.get('ST') else None
    test(
        workdir=W,
        tag_categories=CATES,
        seen_tag_keys=ST,
    )
