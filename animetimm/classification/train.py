import json
import os
import random
from functools import partial
from pprint import pformat
from typing import Optional

import click
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from ditk import logging
from hbutils.random import global_seed
from hbutils.string import plural_word
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
from torch.optim import lr_scheduler
from tqdm import tqdm

from animetimm.utils import GLOBAL_CONTEXT_SETTINGS, print_version, parse_key_value
from .dataset import load_tags, load_dataloader
from .loss import FocalLoss
from ..dataset import load_pretrained_tag
from ..model import Model
from ..session import TrainSession


def train(
        workdir: str,
        dataset_repo_id: str,
        timm_model_name: str,
        tag_key: str,
        num_workers: int = 16,
        max_epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-3,
        key_metric: str = 'top-5',
        seed: Optional[int] = 0,
        eval_epoch: int = 1,
        eval_threshold: float = 0.4,
        model_args: Optional[dict] = None,
        pretrained_cfg: Optional[dict] = None,
        noise_level: int = 2,
        rotation_ratio: float = 0.25,
        cutout_max_pct: float = 0.0,
        cutout_patches: int = 0,
        random_resize_method: bool = True,
        pre_align: bool = True,
        align_size: int = 512,
        image_key: str = 'webp',
        cof: float = 1.0,
        filters: Optional[dict] = None,
        grayscale_prob: float = 0.0,
):
    accelerator = Accelerator(
        # mixed_precision=self.cfgs.mixed_precision,
        step_scheduler_with_optimizer=False,
    )

    if seed is None:
        seed = random.randint(0, (1 << 31) - 1)
    blist = [seed]
    broadcast_object_list(blist, from_process=0)
    seed = blist[0] + accelerator.process_index
    # native random, numpy, torch and faker's seeds are includes
    # if you need to register more library for seeding, see:
    # https://hansbug.github.io/hbutils/main/api_doc/random/state.html#register-random-source
    logging.info(f'Globally set the random seed {seed!r} in process #{accelerator.process_index}.')
    global_seed(seed)

    os.makedirs(workdir, exist_ok=True)

    filters = dict(filters or {})
    tags_info = load_tags(repo_id=dataset_repo_id, filters=filters, cof=cof)
    tags_info.df.to_csv(os.path.join(workdir, 'tags.csv'), index=False)
    checkpoints = os.path.join(workdir, 'checkpoints')
    last_ckpt_zip_file = os.path.join(checkpoints, 'last.zip')
    model_args = dict(model_args or {})
    model_args = {**model_args}
    pretrained_cfg = dict(pretrained_cfg or {})
    pretrained_cfg = {'crop_pct': 1.0, 'test_crop_pct': 1.0, **pretrained_cfg}
    if os.path.exists(last_ckpt_zip_file):
        if accelerator.is_main_process:
            logging.info(f'Loading last checkpoint from {last_ckpt_zip_file!r} ...')
        model, meta, metrics = Model.load_from_zip(last_ckpt_zip_file)
        if model.model_name != timm_model_name:
            raise RuntimeError(f'Model name not match with the previous checkpoint '
                               f'({timm_model_name!r} vs {model.model_name}), '
                               f'if you insist on opening another training task, please use another workdir.')
        if model.tags != tags_info.tags:
            raise RuntimeError(f'Tag list not match with the previous checkpoint, '
                               f'if you insist on opening another training task, please use another workdir.')
        # if model.model_args != model_args:
        #     raise RuntimeError(f'Model cfgs not match with the previous checkpoint '
        #                        f'({model_args!r} vs {model.model_args}), '
        #                        f'if you insist on opening another training task, please use another workdir.')
        # if model.pretrained_cfg != pretrained_cfg:
        #     raise RuntimeError(f'Pretrained cfgs not match with the previous checkpoint '
        #                        f'({pretrained_cfg!r} vs {model.pretrained_cfg}), '
        #                        f'if you insist on opening another training task, please use another workdir.')
        previous_epoch = meta['step']
        if accelerator.is_main_process:
            logging.info(f'Resume from epoch {previous_epoch!r}.')
    else:
        if accelerator.is_main_process:
            logging.info(f'No last checkpoint found, initialize {timm_model_name!r} model '
                         f'with {plural_word(len(tags_info.tags), "tag")}.')
        model = Model.new(
            model_name=timm_model_name,
            tags=tags_info.tags,
            pretrained=True,
            pretrained_cfg=pretrained_cfg,
            model_args=model_args,
        )
        previous_epoch = 0

    model: Model
    pretrained_tag = load_pretrained_tag(dataset_repo_id)
    logging.info(f'Pretrained tag {pretrained_tag!r} found for dataset {dataset_repo_id!r}.')
    model.pretrained_tag = pretrained_tag
    previous_epoch: int
    train_cfg = {
        'batch_size': batch_size,
        'max_epochs': max_epochs,
        'seed': seed,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'key_metric': key_metric,
        'processes': accelerator.num_processes,
        'eval_threshold': eval_threshold,
        'noise_level': noise_level,
        'rotation_ratio': rotation_ratio,
        'cutout_max_pct': cutout_max_pct,
        'cutout_patches': cutout_patches,
        'random_resize_method': random_resize_method,
        'pre_align': pre_align,
        'align_size': align_size,
        'dataset': dataset_repo_id,
        **model_args,
        'pretrained_tag': pretrained_tag,
        'image_key': image_key,
        'tag_key': tag_key,
        'filters': filters or {},
        'cof': cof,
        'grayscale_prob': grayscale_prob,
    }
    if accelerator.is_main_process:
        logging.info(f'Training configurations: {train_cfg!r}.')
        with open(os.path.join(workdir, 'meta.json'), 'w') as f:
            json.dump({
                'model_name': model.model_name,
                'tags': tags_info.tags,
                'model_args': model.model_args,
                'pretrained_cfg': model.pretrained_cfg,
                'train': train_cfg,
            }, f, indent=4, ensure_ascii=False, sort_keys=True)

    module = model.module
    train_dataloader = load_dataloader(
        repo_id=dataset_repo_id,
        model=module,
        split='train',
        batch_size=batch_size,
        num_workers=num_workers,
        noise_level=noise_level,
        rotation_ratio=rotation_ratio,
        cutout_max_pct=cutout_max_pct,
        cutout_patches=cutout_patches,
        random_resize_method=random_resize_method,
        pre_align=pre_align,
        align_size=align_size,
        is_main_process=accelerator.is_main_process,
        image_key=image_key,
        tag_key=tag_key,
        tag_filters=filters,
    )
    eval_dataloader = load_dataloader(
        repo_id=dataset_repo_id,
        model=module,
        split='validation',
        batch_size=batch_size,
        num_workers=num_workers,
        pre_align=pre_align,
        align_size=align_size,
        is_main_process=accelerator.is_main_process,
        image_key=image_key,
        tag_key=tag_key,
        tag_filters=filters,
    )

    loss_fn = FocalLoss(reduction='none', num_classes=len(tags_info.tags), weight=tags_info.weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, module.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    module, optimizer, train_dataloader, eval_dataloader, loss_fn = \
        accelerator.prepare(module, optimizer, train_dataloader, eval_dataloader, loss_fn)

    # scheduler do not need to get prepared
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=len(train_dataloader),
        epochs=max_epochs,
        pct_start=0.15,
        final_div_factor=20.,
    )
    # start from previous LR
    for _ in range(previous_epoch * len(train_dataloader)):
        scheduler.step()

    if accelerator.is_main_process:
        logging.info(f'Model Class: {type(module)!r}')
        session = TrainSession(
            workdir, key_metric=key_metric,
            extra_metadata={
                **{f'train/{key}': value for key, value in train_cfg.items()},
            },
            hyperparams=train_cfg,
            project=f'{dataset_repo_id}',
        )
        logging.info('Training start!')

    accelerator.wait_for_everyone()

    for epoch in range(previous_epoch + 1, max_epochs + 1):
        if accelerator.is_local_main_process:
            logging.info(f'Training for epoch {epoch!r}')
        module.train()
        train_lr = scheduler.get_last_lr()[0]
        train_loss = 0.0
        train_total = 0
        train_top1, train_top5 = 0, 0

        labs, preds = [], []
        for i, (inputs, labels_) in enumerate(tqdm(train_dataloader, disable=not accelerator.is_main_process)):
            inputs = inputs.float()
            labels_ = labels_

            optimizer.zero_grad()
            outputs = module(inputs)
            train_total += labels_.shape[0]

            with torch.no_grad():
                as_ = torch.argsort(outputs, dim=-1)
                train_top1 += (as_[:, -1] == labels_).sum().detach().cpu().item()
                as_top5 = as_[:, -5:]
                for t5, expected in zip(as_top5.detach().cpu().tolist(), labels_.detach().cpu().tolist()):
                    if expected in t5:
                        train_top5 += 1

                labs.append(labels_.clone().detach())
                preds.append(torch.argmax(outputs, dim=-1).detach())

            loss = loss_fn(outputs, labels_).sum()
            accelerator.backward(loss)
            optimizer.step()
            train_loss += loss.item()
            scheduler.step()

        accelerator.wait_for_everyone()

        with torch.no_grad():
            labs = torch.concat(labs)
            preds = torch.concat(preds)
            train_loss = accelerator.gather(
                torch.tensor([train_loss], device=accelerator.device)).sum().detach().cpu().item()
            train_total = accelerator.gather(
                torch.tensor([train_total], device=accelerator.device)).sum().detach().cpu().item()

            labs = accelerator.gather(labs).detach().cpu().numpy()
            preds = accelerator.gather(preds).detach().cpu().numpy()

            train_top1 = accelerator.gather(
                torch.tensor([train_top1], device=accelerator.device)).sum().detach().cpu().item()
            train_top5 = accelerator.gather(
                torch.tensor([train_top5], device=accelerator.device)).sum().detach().cpu().item()

        if accelerator.is_main_process:
            with torch.no_grad():
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
            session.tb_train_log(
                global_step=epoch,
                metrics={
                    'loss': train_loss / train_total,
                    'top-1': train_top1 * 1.0 / train_total,
                    'top-5': train_top5 * 1.0 / train_total,
                    'micro_f1': micro_f1,
                    'micro_precision': micro_precision,
                    'micro_recall': micro_recall,
                    'macro_f1': macro_f1,
                    'macro_precision': macro_precision,
                    'macro_recall': macro_recall,
                    'learning_rate': train_lr,
                    'details': df_tags_details,
                }
            )

        if epoch % eval_epoch == 0:
            module.eval()

            with torch.no_grad():
                eval_loss = 0.0
                eval_total = 0
                eval_top1, eval_top5 = 0, 0

                labs, preds = [], []
                for i, (inputs, labels_) in enumerate(tqdm(eval_dataloader, disable=not accelerator.is_main_process)):
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

                    loss = loss_fn(outputs, labels_).sum()
                    eval_loss += loss.item()

                accelerator.wait_for_everyone()

                labs = torch.concat(labs)
                preds = torch.concat(preds)
                eval_loss = accelerator.gather(
                    torch.tensor([eval_loss], device=accelerator.device)).sum().detach().cpu().item()
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
                    session.tb_eval_log(
                        global_step=epoch,
                        model=model,
                        metrics={
                            'loss': eval_loss / eval_total,
                            'top-1': eval_top1 * 1.0 / eval_total,
                            'top-5': eval_top5 * 1.0 / eval_total,
                            'micro_f1': micro_f1,
                            'micro_precision': micro_precision,
                            'micro_recall': micro_recall,
                            'macro_f1': macro_f1,
                            'macro_precision': macro_precision,
                            'macro_recall': macro_recall,
                            'details': df_tags_details,
                        }
                    )


@click.command(context_settings={**GLOBAL_CONTEXT_SETTINGS}, help="Training single-label classification models.")
@click.option('-v', '--version', is_flag=True,
              callback=partial(print_version, 'animetimm.classification.train'), expose_value=False, is_eager=True)
@click.option('--tag-key', '-tk', required=True, help='Tag key in webdataset', show_default=True)
@click.option('--dataset-repo-id', '-ds', required=True, help='Dataset repository to use. '
                                                              '-s option will be ignored when -ds is used.',
              show_default=True)
@click.option('--max-epochs', '-mep', default=100, type=int, help='Maximum number of epochs', show_default=True)
@click.option('--model-name', '-m', default='caformer_s36.sail_in22k_ft_in1k_384', help='Model name', show_default=True)
@click.option('--size', type=int, help='Image size', show_default=True)
@click.option('--num-workers', '-nw', default=32, type=int, help='Number of workers', show_default=True)
@click.option('--batch-size', '-bs', default=32, type=int, help='Batch size', show_default=True)
@click.option('--learning-rate', '-lr', default=2e-4, type=float, help='Learning rate', show_default=True)
@click.option('--weight-decay', '-wd', default=1e-3, type=float, help='Weight decay', show_default=True)
@click.option('--key-metric', '-km', default='top-5', help='Key metric for evaluation', show_default=True)
@click.option('--seed', type=int, default=0, help='Random seed', show_default=True)
@click.option('--eval-epoch', '-ee', default=1, type=int, help='Evaluation epoch interval', show_default=True)
@click.option('--eval-threshold', '-et', default=0.4, type=float, help='Evaluation threshold', show_default=True)
@click.option('--noise-level', '-nl', default=2, type=int, help='Noise level', show_default=True)
@click.option('--rotation-ratio', '-rr', default=0.25, type=float, help='Rotation ratio', show_default=True)
@click.option('--cutout-max-pct', '-cmp', default=0.0, type=float, help='Cutout max percentage', show_default=True)
@click.option('--cutout-patches', '-cp', default=0, type=int, help='Cutout patches', show_default=True)
@click.option('--random-resize-method/--no-random-resize-method', default=True, help='Random resize method',
              show_default=True)
@click.option('--pre-align/--no-pre-align', default=True, help='Pre-align', show_default=True)
@click.option('--align-size', '-as', default=512, type=int, help='Align size', show_default=True)
@click.option('--drop-path-rate', '-dpr', default=0.4, type=float, help='Drop path rate', show_default=True)
@click.option('--workdir', '-w', default=None, type=str, help='Workdir to save training data', show_default=True)
@click.option('--suffix', '-sf', 'suffix', default='', type=str, help='Work directory suffix', show_default=True)
@click.option('--image_key', '-ik', default='webp', type=str, help='Image key in webdataset.', show_default=True)
@click.option('--model-arg', '-ma', multiple=True, callback=parse_key_value,
              help='Additional model arguments in format KEY=VALUE. Types are auto-detected.\n'
                   'Use KEY:str=VALUE to force string type.\n'
                   'Supported type hints: str, int, float, bool, none, list.\n'
                   'Examples:\n'
                   '--model-arg depth=12\n'
                   '--model-arg embed_dim=768\n'
                   '--model-arg use_cls_token:bool=true\n'
                   '--model-arg name:str=123\n'
                   '--model-arg layers:list=1,2,3',
              show_default=True)
@click.option('--filters', multiple=True, callback=parse_key_value,
              help='Add filters to tag list. The format is similar to --model-arg.',
              show_default=True)
@click.option('--cof', '-cf', default=1.0, type=float, help='Co-efficient of class weights.', show_default=True)
@click.option('--grayscale_prob', '-gp', default=0.0, type=float, help='Grayscale probability when training',
              show_default=True)
def cli(tag_key, dataset_repo_id, max_epochs, model_name, size, num_workers, batch_size, learning_rate, weight_decay,
        key_metric, seed, eval_epoch, eval_threshold, noise_level, rotation_ratio,
        cutout_max_pct, cutout_patches, random_resize_method, pre_align, align_size,
        drop_path_rate, workdir, model_arg, image_key, filters, cof, grayscale_prob, suffix):
    logging.try_init_root(logging.INFO)

    rmn = model_name.replace('/', '_').replace(':', '_').replace('\\', '_')
    model_args = {
        'drop_path_rate': drop_path_rate,
    }
    if model_arg:
        model_args.update(model_arg)
    if size:
        model_args['img_size'] = size
    logging.info(f'Model args to use:\n{pformat(model_args)}')
    logging.info(f'Filters for tag list:\n{pformat(filters)}')

    size_suffix = f"_s{size}" if size else ""
    pre_align_mark = f'_p{align_size}' if pre_align else ''
    pretrained_tag = load_pretrained_tag(dataset_repo_id)
    workdir = workdir or f'runs/{rmn}_{pretrained_tag}_bs{batch_size}{pre_align_mark}' \
                         f'_d{drop_path_rate}_mep{max_epochs}{size_suffix}{f"_{suffix}" if suffix else ""}'
    logging.info(f'Training on dataset {dataset_repo_id!r}, workdir: {workdir!r}.')

    train(
        workdir=workdir,
        tag_key=tag_key,
        dataset_repo_id=dataset_repo_id,
        timm_model_name=model_name,
        num_workers=num_workers,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        key_metric=key_metric,
        seed=seed,
        eval_epoch=eval_epoch,
        eval_threshold=eval_threshold,
        model_args=model_args,
        pretrained_cfg=None,
        noise_level=noise_level,
        rotation_ratio=rotation_ratio,
        cutout_max_pct=cutout_max_pct,
        cutout_patches=cutout_patches,
        random_resize_method=random_resize_method,
        pre_align=pre_align,
        align_size=align_size,
        max_epochs=max_epochs,
        image_key=image_key,
        filters=filters,
        cof=cof,
        grayscale_prob=grayscale_prob,
    )


if __name__ == '__main__':
    cli()
