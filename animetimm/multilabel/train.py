import json
import os
import random
from functools import partial
from pprint import pformat
from typing import Optional, Sequence, List, Tuple

import click
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from ditk import logging
from hbutils.random import global_seed
from hbutils.string import plural_word
from torch.nn import BCEWithLogitsLoss
from torch.optim import lr_scheduler
from tqdm import tqdm

from .dataset import load_tags, load_dataloader
from .metrics import mcc, f1score, precision, recall
from ..dataset import load_pretrained_tag
from ..model import Model
from ..session import TrainSession
from ..utils import GLOBAL_CONTEXT_SETTINGS, print_version, parse_key_value, parse_tuple

_DEFAULT_BETAS = (0.9, 0.999)


def train(
        workdir: str,
        dataset_repo_id: str,
        timm_model_name: str,
        num_workers: int = 16,
        max_epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-3,
        key_metric: str = 'macro_f1',
        seed: Optional[int] = 0,
        eval_epoch: int = 1,
        eval_threshold: float = 0.4,
        model_args: Optional[dict] = None,
        pretrained_cfg: Optional[dict] = None,
        noise_level: int = 2,
        rotation_ratio: float = 0.0,
        mixup_alpha: float = 0.6,
        cutout_max_pct: float = 0.0,
        cutout_patches: int = 0,
        random_resize_method: bool = True,
        pre_align: bool = True,
        align_size: int = 512,
        tag_categories: Optional[Sequence[int]] = None,
        seen_tag_keys: Optional[List[str]] = None,
        image_key: str = 'webp',
        use_pretrained_weight: bool = True,
        adam_betas: Optional[Tuple[float, float]] = None,
        use_normalize: bool = False,
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

    tags_info = load_tags(repo_id=dataset_repo_id, categories=tag_categories)
    if accelerator.is_main_process:
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
            pretrained=use_pretrained_weight,
            pretrained_cfg=pretrained_cfg,
            model_args=model_args,
        )
        previous_epoch = 0

    model: Model
    pretrained_tag = load_pretrained_tag(dataset_repo_id)
    logging.info(f'Pretrained tag {pretrained_tag!r} found for dataset {dataset_repo_id!r}.')
    model.pretrained_tag = pretrained_tag
    previous_epoch: int
    adam_betas = adam_betas or _DEFAULT_BETAS
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
        'mixup_alpha': mixup_alpha,
        'cutout_max_pct': cutout_max_pct,
        'cutout_patches': cutout_patches,
        'random_resize_method': random_resize_method,
        'pre_align': pre_align,
        'align_size': align_size,
        'dataset': dataset_repo_id,
        **model_args,
        'pretrained_tag': pretrained_tag,
        'tag_categories': tag_categories,
        'seen_tag_keys': seen_tag_keys,
        'image_key': image_key,
        'use_pretrained_weight': use_pretrained_weight,
        'adam_betas': adam_betas,
        'use_normalize': use_normalize,
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
        mixup_alpha=mixup_alpha,
        cutout_max_pct=cutout_max_pct,
        cutout_patches=cutout_patches,
        random_resize_method=random_resize_method,
        pre_align=pre_align,
        align_size=align_size,
        is_main_process=accelerator.is_main_process,
        categories=tag_categories,
        seen_tag_keys=seen_tag_keys,
        image_key=image_key,
        use_normalize=use_normalize,
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
        categories=tag_categories,
        seen_tag_keys=seen_tag_keys,
        image_key=image_key,
        use_normalize=use_normalize,
    )

    loss_fn = BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, module.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=tuple(adam_betas),
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

    label_weights = torch.from_numpy(tags_info.weights).to(accelerator.device)
    for epoch in range(previous_epoch + 1, max_epochs + 1):
        if accelerator.is_local_main_process:
            logging.info(f'Training for epoch {epoch!r}')
        module.train()
        train_lr = scheduler.get_last_lr()[0]
        train_loss = 0.0
        train_total = 0

        micro_tp = torch.zeros((1,), device=accelerator.device)
        micro_fp = torch.zeros((1,), device=accelerator.device)
        micro_tn = torch.zeros((1,), device=accelerator.device)
        micro_fn = torch.zeros((1,), device=accelerator.device)

        macro_tp = torch.zeros((1, len(tags_info.tags),), device=accelerator.device)
        macro_fp = torch.zeros((1, len(tags_info.tags),), device=accelerator.device)
        macro_tn = torch.zeros((1, len(tags_info.tags),), device=accelerator.device)
        macro_fn = torch.zeros((1, len(tags_info.tags),), device=accelerator.device)

        for i, (inputs, labels_) in enumerate(tqdm(train_dataloader, disable=not accelerator.is_local_main_process,
                                                   desc=f'Train on Rank #{accelerator.process_index}')):
            inputs = inputs.float()
            labels_ = labels_

            optimizer.zero_grad()
            outputs = module(inputs)
            train_total += labels_.shape[0]

            with torch.no_grad():
                labels = labels_ > eval_threshold
                preds = torch.sigmoid(outputs) > eval_threshold
                micro_tp += ((preds == 1) & (labels == 1)).sum().item()
                micro_fp += ((preds == 1) & (labels == 0)).sum().item()
                micro_tn += ((preds == 0) & (labels == 0)).sum().item()
                micro_fn += ((preds == 0) & (labels == 1)).sum().item()
                macro_tp += ((preds == 1) & (labels == 1)).sum(dim=0)
                macro_fp += ((preds == 1) & (labels == 0)).sum(dim=0)
                macro_tn += ((preds == 0) & (labels == 0)).sum(dim=0)
                macro_fn += ((preds == 0) & (labels == 1)).sum(dim=0)

            loss = loss_fn(outputs, labels_)
            loss = (loss * label_weights).sum()
            accelerator.backward(loss)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            scheduler.step()

        accelerator.wait_for_everyone()

        with torch.no_grad():
            train_loss = accelerator.gather(
                torch.tensor([train_loss], device=accelerator.device)).sum().detach().cpu().item()
            train_total = accelerator.gather(
                torch.tensor([train_total], device=accelerator.device)).sum().detach().cpu().item()
            micro_tp = accelerator.gather(micro_tp).sum(dim=0)
            micro_fp = accelerator.gather(micro_fp).sum(dim=0)
            micro_tn = accelerator.gather(micro_tn).sum(dim=0)
            micro_fn = accelerator.gather(micro_fn).sum(dim=0)

            macro_tp = accelerator.gather(macro_tp).sum(dim=0)
            macro_fp = accelerator.gather(macro_fp).sum(dim=0)
            macro_tn = accelerator.gather(macro_tn).sum(dim=0)
            macro_fn = accelerator.gather(macro_fn).sum(dim=0)

        if accelerator.is_main_process:
            with torch.no_grad():
                micro_mcc = mcc(micro_tp, micro_fp, micro_tn, micro_fn).detach().cpu().item()
                micro_f1 = f1score(micro_tp, micro_fp, micro_tn, micro_fn).detach().cpu().item()
                micro_precision = precision(micro_tp, micro_fp, micro_tn, micro_fn).detach().cpu().item()
                micro_recall = recall(micro_tp, micro_fp, micro_tn, micro_fn).detach().cpu().item()

                macro_mcc = mcc(macro_tp, macro_fp, macro_tn, macro_fn).detach().cpu().item()
                macro_f1 = f1score(macro_tp, macro_fp, macro_tn, macro_fn).detach().cpu().item()
                macro_precision = precision(macro_tp, macro_fp, macro_tn, macro_fn).detach().cpu().item()
                macro_recall = recall(macro_tp, macro_fp, macro_tn, macro_fn).detach().cpu().item()

                macro_mcc_lst = mcc(macro_tp, macro_fp, macro_tn, macro_fn, mean=False).detach().cpu().tolist()
                macro_f1_lst = f1score(macro_tp, macro_fp, macro_tn, macro_fn, mean=False).detach().cpu().tolist()
                macro_precision_lst = precision(macro_tp, macro_fp, macro_tn, macro_fn,
                                                mean=False).detach().cpu().tolist()
                macro_recall_lst = recall(macro_tp, macro_fp, macro_tn, macro_fn, mean=False).detach().cpu().tolist()

            df_tags_details = pd.DataFrame({
                **{name: tags_info.df[name] for name in tags_info.df.columns},
                'mcc': macro_mcc_lst,
                'f1': macro_f1_lst,
                'precision': macro_precision_lst,
                'recall': macro_recall_lst,
            })
            session.tb_train_log(
                global_step=epoch,
                metrics={
                    'loss': train_loss / train_total / len(tags_info.tags),
                    'micro_mcc': micro_mcc,
                    'micro_f1': micro_f1,
                    'micro_precision': micro_precision,
                    'micro_recall': micro_recall,
                    'macro_mcc': macro_mcc,
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

                micro_tp = torch.zeros((1,), device=accelerator.device)
                micro_fp = torch.zeros((1,), device=accelerator.device)
                micro_tn = torch.zeros((1,), device=accelerator.device)
                micro_fn = torch.zeros((1,), device=accelerator.device)

                macro_tp = torch.zeros((1, len(tags_info.tags),), device=accelerator.device)
                macro_fp = torch.zeros((1, len(tags_info.tags),), device=accelerator.device)
                macro_tn = torch.zeros((1, len(tags_info.tags),), device=accelerator.device)
                macro_fn = torch.zeros((1, len(tags_info.tags),), device=accelerator.device)

                for i, (inputs, labels_) in enumerate(
                        tqdm(eval_dataloader, disable=not accelerator.is_local_main_process,
                             desc=f'Eval on Rank #{accelerator.process_index}')):
                    inputs = inputs.float()
                    labels_ = labels_

                    outputs = module(inputs)
                    eval_total += labels_.shape[0]

                    labels = labels_ > eval_threshold
                    preds = torch.sigmoid(outputs) > eval_threshold
                    micro_tp += ((preds == 1) & (labels == 1)).sum().item()
                    micro_fp += ((preds == 1) & (labels == 0)).sum().item()
                    micro_tn += ((preds == 0) & (labels == 0)).sum().item()
                    micro_fn += ((preds == 0) & (labels == 1)).sum().item()
                    macro_tp += ((preds == 1) & (labels == 1)).sum(dim=0)
                    macro_fp += ((preds == 1) & (labels == 0)).sum(dim=0)
                    macro_tn += ((preds == 0) & (labels == 0)).sum(dim=0)
                    macro_fn += ((preds == 0) & (labels == 1)).sum(dim=0)

                    loss = loss_fn(outputs, labels_)
                    loss = (loss * label_weights).sum()
                    eval_loss += loss.item() * inputs.size(0)

                accelerator.wait_for_everyone()

                eval_loss = accelerator.gather(
                    torch.tensor([eval_loss], device=accelerator.device)).sum().detach().cpu().item()
                eval_total = accelerator.gather(
                    torch.tensor([eval_total], device=accelerator.device)).sum().detach().cpu().item()
                micro_tp = accelerator.gather(micro_tp).sum(dim=0)
                micro_fp = accelerator.gather(micro_fp).sum(dim=0)
                micro_tn = accelerator.gather(micro_tn).sum(dim=0)
                micro_fn = accelerator.gather(micro_fn).sum(dim=0)

                macro_tp = accelerator.gather(macro_tp).sum(dim=0)
                macro_fp = accelerator.gather(macro_fp).sum(dim=0)
                macro_tn = accelerator.gather(macro_tn).sum(dim=0)
                macro_fn = accelerator.gather(macro_fn).sum(dim=0)

                if accelerator.is_main_process:
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
                    })
                    session.tb_eval_log(
                        global_step=epoch,
                        model=model,
                        metrics={
                            'loss': eval_loss / eval_total / len(tags_info.tags),
                            'micro_mcc': micro_mcc,
                            'micro_f1': micro_f1,
                            'micro_precision': micro_precision,
                            'micro_recall': micro_recall,
                            'macro_mcc': macro_mcc,
                            'macro_f1': macro_f1,
                            'macro_precision': macro_precision,
                            'macro_recall': macro_recall,
                            'learning_rate': train_lr,
                            'details': df_tags_details,
                        }
                    )


@click.command(context_settings={**GLOBAL_CONTEXT_SETTINGS}, help="Training multilabel taggers.")
@click.option('-v', '--version', is_flag=True,
              callback=partial(print_version, 'animetimm.multilabel.train'), expose_value=False, is_eager=True)
@click.option('--dataset-repo-id', '-ds', default='animetimm/danbooru-wdtagger-v4-w640-ws-full',
              help='Dataset repository to use.',
              show_default=True)
@click.option('--max-epochs', '-mep', default=100, type=int, help='Maximum number of epochs', show_default=True)
@click.option('--model-name', '-m', default='caformer_s36.sail_in22k_ft_in1k_384', help='Model name', show_default=True)
@click.option('--size', type=int, help='Image size', show_default=True)
@click.option('--num-workers', '-nw', default=32, type=int, help='Number of workers', show_default=True)
@click.option('--batch-size', '-bs', default=32, type=int, help='Batch size', show_default=True)
@click.option('--learning-rate', '-lr', default=2e-4, type=float, help='Learning rate', show_default=True)
@click.option('--weight-decay', '-wd', default=1e-3, type=float, help='Weight decay', show_default=True)
@click.option('--key-metric', '-km', default='macro_f1', help='Key metric for evaluation', show_default=True)
@click.option('--seed', type=int, default=0, help='Random seed', show_default=True)
@click.option('--eval-epoch', '-ee', default=1, type=int, help='Evaluation epoch interval', show_default=True)
@click.option('--eval-threshold', '-et', default=0.4, type=float, help='Evaluation threshold', show_default=True)
@click.option('--noise-level', '-nl', default=2, type=int, help='Noise level', show_default=True)
@click.option('--rotation-ratio', '-rr', default=0.0, type=float, help='Rotation ratio', show_default=True)
@click.option('--mixup-alpha', default=0.6, type=float, help='Mixup alpha', show_default=True)
@click.option('--cutout-max-pct', '-cmp', default=0.0, type=float, help='Cutout max percentage', show_default=True)
@click.option('--cutout-patches', '-cp', default=0, type=int, help='Cutout patches', show_default=True)
@click.option('--random-resize-method/--no-random-resize-method', default=True, help='Random resize method',
              show_default=True)
@click.option('--pre-align/--no-pre-align', default=True, help='Pre-align', show_default=True)
@click.option('--align-size', '-as', default=512, type=int, help='Align size', show_default=True)
@click.option('--use-pretrained/--no-pretrained', 'use_pretrained_weight', default=True,
              help='Use pretrained weights. Disable pretrained weights when you '
                   'have to change some architectures', show_default=True)
@click.option('--tag-categories', '-tc', multiple=True, type=int, help='Tag categories (multiple)', show_default=True)
@click.option('--seen-tag-keys', '-stk', multiple=True, help='Seen tag keys (multiple)', show_default=True)
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
@click.option('--adam-betas', '--betas', 'adam_betas', callback=parse_tuple, default=None,
              help='Beta values for adam-like optimizers.', show_default=True)
@click.option('--use_normalize', '-un', 'use_normalize', is_flag=True, type=bool, default=False,
              help='Use normalize value from dataset repository.', show_default=True)
def cli(dataset_repo_id, max_epochs, model_name, size, num_workers, batch_size, learning_rate, weight_decay,
        key_metric, seed, eval_epoch, eval_threshold, noise_level, rotation_ratio, mixup_alpha,
        cutout_max_pct, cutout_patches, random_resize_method, pre_align, align_size,
        tag_categories, seen_tag_keys, drop_path_rate, workdir, model_arg, image_key, suffix, use_pretrained_weight,
        adam_betas, use_normalize):
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

    size_suffix = f"_s{size}" if size else ""
    pre_align_mark = f'_p{align_size}' if pre_align else ''
    pretrained_tag = load_pretrained_tag(dataset_repo_id)
    workdir = workdir or f'runs/{rmn}_{pretrained_tag}_bs{batch_size}{pre_align_mark}' \
                         f'_d{drop_path_rate}_mep{max_epochs}{size_suffix}{f"_{suffix}" if suffix else ""}' \
                         f'{"_norm" if use_normalize else ""}'
    logging.info(f'Training on dataset {dataset_repo_id!r}, workdir: {workdir!r}.')

    tag_categories_seq = list(tag_categories) if tag_categories else None
    seen_tag_keys_list = list(seen_tag_keys) if seen_tag_keys else None

    train(
        workdir=workdir,
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
        mixup_alpha=mixup_alpha,
        cutout_max_pct=cutout_max_pct,
        cutout_patches=cutout_patches,
        random_resize_method=random_resize_method,
        pre_align=pre_align,
        align_size=align_size,
        tag_categories=tag_categories_seq,
        seen_tag_keys=seen_tag_keys_list,
        max_epochs=max_epochs,
        image_key=image_key,
        use_pretrained_weight=use_pretrained_weight,
        adam_betas=adam_betas,
        use_normalize=use_normalize,
    )


if __name__ == '__main__':
    cli()

# if __name__ == '__main__':
#     logging.try_init_root(logging.INFO)
#     set_name = str(os.environ.get('S', '150k') or '150k')
#     max_epoch = int(os.environ.get('MEP', '100') or '100')
#     mn = str(os.environ.get('M', 'caformer_s36.sail_in22k_ft_in1k_384') or 'caformer_s36.sail_in22k_ft_in1k_384')
#     rmn = mn.replace('/', '_').replace(':', '_').replace('\\', '_')
#     size = os.environ.get('SIZE')
#     num_workers = int(os.environ.get('NW', 32) or 32)
#     if size:
#         size = int(size)
#     batch_size = int(os.environ.get('BS', 64) or 64)
#     model_args = dict(
#         drop_path_rate=0.4,
#     )
#     if size:
#         model_args['img_size'] = size
#     train(
#         workdir=f'runs/{rmn}_{set_name}_bs{batch_size}_p512x_d0.4_mep{max_epoch}_s{size or ""}',
#         dataset_repo_id=f'animetimm/danbooru-wdtagger-v4-w640-ws-{set_name}',
#         timm_model_name=mn,
#         num_workers=num_workers,
#         batch_size=batch_size,
#         learning_rate=2e-4,
#         noise_level=2,
#         mixup_alpha=0.6,
#         cutout_patches=0,
#         cutout_max_pct=0.0,
#         rotation_ratio=0.0,
#         model_args=model_args,
#         max_epochs=max_epoch,
#     )
