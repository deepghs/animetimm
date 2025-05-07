import json
import os
import random
from typing import Optional

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

from animetimm.model import Model
from animetimm.multilabel.metrics import mcc, f1score, precision, recall
from animetimm.session import TrainSession
from .dataset import load_tags, load_dataloader


def train(
        workdir: str,
        dataset_repo_id: str,
        timm_model_name: str,
        num_workers: int = 8,
        max_epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        key_metric: str = 'macro_f1',
        seed: Optional[int] = 0,
        eval_epoch: int = 1,
        eval_threshold: float = 0.4,
        model_cfg: Optional[dict] = None,
        pretrained_cfg: Optional[dict] = None,
        noise_level: int = 1,
        rotation_ratio: float = 0.0,
        mixup_alpha: float = 0.6,
        cutout_max_pct: float = 0.25,
        cutout_patches: int = 1,
        random_resize_method: bool = True,
        pre_align: bool = True,
        align_size: int = 512
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

    tags_info = load_tags(repo_id=dataset_repo_id)
    checkpoints = os.path.join(workdir, 'checkpoints')
    last_ckpt_zip_file = os.path.join(checkpoints, 'last.zip')
    model_cfg = dict(model_cfg or {})
    pretrained_cfg = dict(pretrained_cfg or {})
    if os.path.exists(last_ckpt_zip_file):
        if accelerator.is_main_process:
            logging.info(f'Loading last checkpoint from {last_ckpt_zip_file!r} ...')
        model, meta, metrics = Model.load_from_zip(last_ckpt_zip_file, device=accelerator.device)
        if model.model_name != timm_model_name:
            raise RuntimeError(f'Model name not match with the previous checkpoint '
                               f'({timm_model_name!r} vs {model.model_name}), '
                               f'if you insist on opening another training task, please use another workdir.')
        if model.tags != tags_info.tags:
            raise RuntimeError(f'Tag list not match with the previous checkpoint, '
                               f'if you insist on opening another training task, please use another workdir.')
        if model.model_cfg != model_cfg:
            raise RuntimeError(f'Model cfgs not match with the previous checkpoint '
                               f'({model_cfg!r} vs {model.model_cfg}), '
                               f'if you insist on opening another training task, please use another workdir.')
        if model.pretrained_cfg != pretrained_cfg:
            raise RuntimeError(f'Pretrained cfgs not match with the previous checkpoint '
                               f'({pretrained_cfg!r} vs {model.pretrained_cfg}), '
                               f'if you insist on opening another training task, please use another workdir.')
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
            model_cfg=model_cfg,
        )
        previous_epoch = 0

    model: Model
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
        'mixup_alpha': mixup_alpha,
        'cutout_max_pct': cutout_max_pct,
        'cutout_patches': cutout_patches,
        'random_resize_method': random_resize_method,
        'pre_align': pre_align,
        'align_size': align_size,
    }
    if accelerator.is_main_process:
        logging.info(f'Training configurations: {train_cfg!r}.')
    with open(os.path.join(workdir, 'meta.json'), 'w') as f:
        json.dump({
            'model_name': model.model_name,
            'tags': tags_info.tags,
            'model_cfg': model.model_cfg,
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
    )
    eval_dataloader = load_dataloader(
        repo_id=dataset_repo_id,
        model=module,
        split='validation',
        batch_size=batch_size,
        num_workers=num_workers,
    )

    loss_fn = BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, module.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    module, optimizer, train_dataloader, test_dataloader, loss_fn = \
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
            }
        )
        logging.info('Training start!')

    label_weights = torch.from_numpy(tags_info.weights).to(accelerator.device)
    for epoch in range(previous_epoch + 1, max_epochs + 1):
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

        for i, (inputs, labels_) in enumerate(tqdm(train_dataloader, disable=not accelerator.is_main_process)):
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

            df_macro = pd.DataFrame({
                **{name: tags_info.df[name] for name in tags_info.df.columns},
                'macro_mcc': macro_mcc_lst,
                'macro_f1': macro_f1_lst,
                'macro_precision': macro_precision_lst,
                'macro_recall': macro_recall_lst,
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
                    'macro': df_macro,
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

                for i, (inputs, labels_) in enumerate(tqdm(test_dataloader, disable=not accelerator.is_main_process)):
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

                    df_macro = pd.DataFrame({
                        **{name: tags_info.df[name] for name in tags_info.df.columns},
                        'macro_mcc': macro_mcc_lst,
                        'macro_f1': macro_f1_lst,
                        'macro_precision': macro_precision_lst,
                        'macro_recall': macro_recall_lst,
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
                            'macro': df_macro,
                        }
                    )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    set_name = str(os.environ.get('S', '150k') or '150k')
    train(
        workdir=f'runs/tiny_experiments_{set_name}_p512',
        dataset_repo_id=f'animetimm/danbooru-wdtagger-v4-w640-ws-{set_name}',
        timm_model_name='caformer_s36.sail_in22k_ft_in1k_384',
        num_workers=32,
        batch_size=64,
        learning_rate=2e-4,
    )
