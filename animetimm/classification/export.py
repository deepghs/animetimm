import glob
import json
import os
import re
import shutil
from functools import partial
from pprint import pformat
from tempfile import TemporaryDirectory
from typing import Optional, Literal, List

import click
import pandas as pd
from PIL import Image
from ditk import logging
from hbutils.encoding import sha3
from hfutils.operate import get_hf_client, upload_directory_as_directory
from hfutils.repository import hf_hub_repo_url
from huggingface_hub import hf_hub_url
from imgutils.preprocess.torchvision import PadToSize, parse_torchvision_transforms
from thop import clever_format
from timm.models._hub import save_for_hf
from torchvision.transforms import Compose

from .augmentation import create_transforms
from .test import test
from ..dataset import load_pretrained_tag
from ..model import Model
from ..onnx import export_model_to_onnx, ExportedONNXNotUniqueError
from ..utils import GLOBAL_CONTEXT_SETTINGS, print_version, is_tensorboard_has_content, \
    VALID_LICENCES, torch_model_profile_via_calflops

_LOG_FILE_PATTERN = re.compile(r'^events\.out\.tfevents\.(?P<timestamp>\d+)\.(?P<machine>[^.]+)\.(?P<extra>[\s\S]+)$')


def export(workdir: str, repo_id: Optional[str] = None,
           visibility: Literal['private', 'public', 'gated', 'manual'] = 'private',
           logfile_anonymous: bool = True, append_tags: Optional[List[str]] = None,
           title: Optional[str] = None, description: Optional[str] = None, license: str = 'mit',
           onnx_opset_version: int = 14, no_onnx_export: bool = False):
    if os.path.exists(os.path.join(workdir, 'test_options.json')):
        with open(os.path.join(workdir, 'test_options.json'), 'r') as f:
            test_config_info = json.load(f)
    else:
        test_config_info = {}
    use_test_size = test_config_info.get('use_test_size', True)

    append_tags = list(append_tags or [])
    hf_client = get_hf_client()
    with TemporaryDirectory() as upload_dir:
        meta_info_file = os.path.join(workdir, 'meta.json')
        logging.info(f'Loading meta from {meta_info_file!r} ...')
        with open(meta_info_file, 'r') as f:
            meta_info = json.load(f)
        meta_info['type'] = 'classification'

        dataset_repo_id = meta_info['train']['dataset']
        checkpoints = os.path.join(workdir, 'checkpoints')
        best_ckpt_zip_file = os.path.join(checkpoints, 'best.zip')
        logging.info(f'Loading model from {best_ckpt_zip_file!r} ...')
        model, meta, metrics = Model.load_from_zip(best_ckpt_zip_file)
        if not use_test_size:
            model.pretrained_cfg['test_input_size'] = model.pretrained_cfg['input_size']
        model.pretrained_cfg['license'] = license
        model.module.pretrained_cfg.update(model.pretrained_cfg)

        model: Model
        pretrained_tag = meta_info['train'].get('pretrained_tag') or load_pretrained_tag(dataset_repo_id)
        logging.info(f'Pretrained tag {pretrained_tag!r} found for dataset {dataset_repo_id!r}.')
        model.pretrained_tag = pretrained_tag

        model_name = '.'.join([model.architecture, pretrained_tag, *append_tags])
        repo_id = repo_id or f'animetimm/{model_name}'
        logging.info(f'Target repository: {repo_id!r}.')
        if not hf_client.repo_exists(repo_id=repo_id, repo_type='model'):
            hf_client.create_repo(repo_id=repo_id, repo_type='model', private=visibility == 'private')
            if visibility == 'gated':
                hf_client.update_repo_settings(repo_id=repo_id, repo_type='model', gated='auto')
            elif visibility == 'manual':
                hf_client.update_repo_settings(repo_id=repo_id, repo_type='model', gated='manual')

        logging.info(f'Dumping as huggingface TIMM format to {upload_dir!r} ...')
        save_for_hf(
            model.module,
            upload_dir,
            model_config={
                'labels': model.tags,
            },
            model_args=model.get_actual_model_args(),
            safe_serialization='both',
        )

        # print(metrics)
        metrics_file = os.path.join(upload_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            metrics_info = {}
            if os.path.exists(os.path.join(workdir, 'test_metrics.json')):
                with open(os.path.join(workdir, 'test_metrics.json'), 'r') as mf:
                    ji = json.load(mf)
                    metrics_info['test'] = ji
            metrics_info['val'] = {
                key.split('/', maxsplit=1)[-1]: value
                for key, value in metrics.items()
                if isinstance(value, (type(None), int, float, str))
            }
            logging.info(f'Writing metrics to {metrics_file!r}:\n{pformat(metrics_info)}')
            json.dump(metrics_info, f, sort_keys=True, ensure_ascii=False, indent=4)

        df_eval_tags: pd.DataFrame = metrics['df_details']
        df_tags = df_eval_tags.rename(columns={'f1': 'val_f1', 'precision': 'val_precision', 'recall': 'val_recall'})
        if os.path.exists(os.path.join(workdir, 'test_tags.csv')):
            df_test_tags = pd.read_csv(os.path.join(workdir, 'test_tags.csv'), keep_default_na=False)
            df_tags['test_f1'] = df_test_tags['f1']
            df_tags['test_precision'] = df_test_tags['precision']
            df_tags['test_recall'] = df_test_tags['recall']
        tags_file = os.path.join(upload_dir, 'selected_tags.csv')
        logging.info(f'Dumping tags with metrics to {tags_file!r}:\n{df_tags}')
        df_tags.to_csv(tags_file, index=False)

        transforms_file = os.path.join(upload_dir, 'preprocess.json')
        logging.info(f'Dumping preprocessors to {transforms_file!r} ...')
        with open(transforms_file, 'w') as f:
            eval_trans = create_transforms(
                timm_model=model.module,
                is_training=False,
                use_test_size=False,
                noise_level=0,
                rotation_ratio=0,
                cutout_patches=0,
                cutout_max_pct=0.0,
                random_resize_method=False,
                pre_align=meta_info['train']['pre_align'],
                align_size=meta_info['train']['align_size'],
                grayscale_prob=False,
            )
            logging.info(f'Eval transform:\n{eval_trans}')

            test_trans = create_transforms(
                timm_model=model.module,
                is_training=False,
                use_test_size=use_test_size,
                noise_level=0,
                rotation_ratio=0,
                cutout_patches=0,
                cutout_max_pct=0.0,
                random_resize_method=False,
                pre_align=meta_info['train']['pre_align'],
                align_size=meta_info['train']['align_size'],
                grayscale_prob=False,
            )
            logging.info(f'Test transform:\n{eval_trans}')

            pre_trans = []
            if meta_info['train']['pre_align']:
                pre_trans.append(PadToSize(size=meta_info['train']['align_size']))
            pre_trans = Compose(pre_trans)
            logging.info(f'Pre transform:\n{pre_trans}')

            trans = {
                'val': parse_torchvision_transforms(eval_trans),
                'test': parse_torchvision_transforms(test_trans),
                'pre': parse_torchvision_transforms(pre_trans),
            }
            json.dump(trans, f, ensure_ascii=False, sort_keys=True, indent=4)

        image = Image.new('RGB', (1024, 1024), 'white')
        dummy_input_test = test_trans(image).unsqueeze(0)
        dummy_input_val = eval_trans(image).unsqueeze(0)
        logging.info(f'Dummy input for model: {dummy_input_test.shape!r}')

        flops, params, macs = torch_model_profile_via_calflops(model=model.module, input_=dummy_input_test)
        meta_info['flops'] = flops
        meta_info['params'] = params
        meta_info['macs'] = macs
        new_meta_file = os.path.join(upload_dir, 'meta.json')
        logging.info(f'Saving metadata to {new_meta_file!r} ...')
        with open(new_meta_file, 'w') as f:
            json.dump(meta_info, f, indent=4, sort_keys=True, ensure_ascii=False)

        if not no_onnx_export:
            onnx_file = os.path.join(upload_dir, 'model.onnx')
            logging.info(f'Dumping to onnx file {onnx_file!r} ...')
            try:
                export_model_to_onnx(
                    model=model,
                    dummy_input=dummy_input_test,
                    onnx_filename=onnx_file,
                    metadata={**meta, 'tags': model.tags},
                    wrap_mode='softmax',
                    verbose=False,
                    opset_version=onnx_opset_version,
                )
            except ExportedONNXNotUniqueError:
                logging.exception('Non-unique exported ONNX files, so onnx uploading will be disabled.')
                no_onnx_export = True

        for logfile in glob.glob(os.path.join(workdir, 'events.out.tfevents.*')):
            if not is_tensorboard_has_content(logfile):
                logging.warning(f'Tensorboard file {logfile!r} is empty, skipped.')
                continue

            logging.info(f'Tensorboard file {logfile!r} found.')
            matching = _LOG_FILE_PATTERN.fullmatch(os.path.basename(logfile))
            assert matching, f'Log file {logfile!r}\'s name not match with pattern {_LOG_FILE_PATTERN.pattern}.'

            timestamp = matching.group('timestamp')
            machine = matching.group('machine')
            if logfile_anonymous:
                machine = sha3(machine.encode(), n=224)
            extra = matching.group('extra')

            final_name = f'events.out.tfevents.{timestamp}.{machine}.{extra}'
            dst_log_file = os.path.join(upload_dir, final_name)
            logging.info(f'Adding log file {logfile!r} to {dst_log_file!r} ...')
            shutil.copyfile(logfile, dst_log_file)

        with open(os.path.join(upload_dir, 'README.md'), 'w') as f:
            base_model_repo_id = model.src_repo_id
            if base_model_repo_id == repo_id:
                base_models = hf_client.repo_info(repo_id=repo_id, repo_type='model').card_data.get(
                    'base_model') or []
                if base_models:
                    base_model_repo_id = base_models[0]

            print(f'---', file=f)
            print(f'tags:', file=f)
            print(f'- image-classification', file=f)
            print(f'- timm', file=f)
            print(f'- transformers', file=f)
            print(f'- animetimm', file=f)
            print(f'- dghs-imgutils', file=f)
            print(f'library_name: timm', file=f)
            print(f'license: {license}', file=f)
            print(f'datasets:', file=f)
            print(f'- {dataset_repo_id}', file=f)
            print(f'base_model:', file=f)
            print(f'- {base_model_repo_id}', file=f)
            print(f'---', file=f)
            print(f'', file=f)

            title = title or f'Anime Classifier {model_name}'
            print(f'# {title}', file=f)
            print(f'', file=f)
            if description:
                print(f'{description}', file=f)
                print(f'', file=f)

            s_flops, s_params, s_macs = clever_format([flops, params, macs], "%.1f")
            print(f'## Model Details', file=f)
            print(f'', file=f)
            print(f'- **Model Type:** Single-Label Image classification / feature backbone', file=f)
            print(f'- **Model Stats:**', file=f)
            print(f'  - Params: {s_params}', file=f)
            print(f'  - FLOPs / MACs: {s_flops} / {s_macs}', file=f)
            print(f'  - Image size: train = {dummy_input_val.shape[-1]} x {dummy_input_val.shape[-2]}, '
                  f'test = {dummy_input_test.shape[-1]} x {dummy_input_test.shape[-2]}', file=f)
            print(f'- **Dataset:** [{dataset_repo_id}]'
                  f'({hf_hub_repo_url(repo_id=dataset_repo_id, repo_type="dataset", endpoint="https://huggingface.co")})',
                  file=f)
            print(f'  - Labels Count: {len(df_tags)}', file=f)
            print(f'', file=f)

            print(f'## Results', file=f)
            print(f'', file=f)
            s_records = [{
                '#': 'Validation',
                'Top-1': f"{metrics_info['val']['top-1'] * 100.0:.2f}%",
                'Top-5': f"{metrics_info['val']['top-5'] * 100.0:.2f}%",
                'Macro (F1/P/R)': '%.3f / %.3f / %.3f' % (
                    metrics_info['val']['macro_f1'],
                    metrics_info['val']['macro_precision'],
                    metrics_info['val']['macro_recall']
                ),
                'Micro (F1/P/R)': '%.3f / %.3f / %.3f' % (
                    metrics_info['val']['micro_f1'],
                    metrics_info['val']['micro_precision'],
                    metrics_info['val']['micro_recall'],
                ),
            }]
            if 'test' in metrics_info:
                s_records.append({
                    '#': 'Test',
                    'Top-1': f"{metrics_info['test']['top-1'] * 100.0:.2f}%",
                    'Top-5': f"{metrics_info['test']['top-5'] * 100.0:.2f}%",
                    'Macro (F1/P/R)': '%.3f / %.3f / %.3f' % (
                        metrics_info['test']['macro_f1'],
                        metrics_info['test']['macro_precision'],
                        metrics_info['test']['macro_recall'],
                    ),
                    'Micro (F1/P/R)': '%.3f / %.3f / %.3f' % (
                        metrics_info['test']['micro_f1'],
                        metrics_info['test']['micro_precision'],
                        metrics_info['test']['micro_recall'],
                    )
                })
            df_s = pd.DataFrame(s_records)
            print(df_s.to_markdown(index=False, stralign='center', numalign='center'), file=f)
            print(f'', file=f)
            print(f'You can find label list in [selected_tags.csv]'
                  f'({hf_hub_url(repo_id=repo_id, repo_type="model", filename="selected_tags.csv", endpoint="https://huggingface.co")}).',
                  file=f)
            print(f'', file=f)

        upload_directory_as_directory(
            repo_id=repo_id,
            repo_type='model',
            local_directory=upload_dir,
            path_in_repo='.',
            message=f'Upload model {repo_id!r}',
            clear=True,
        )


@click.command(context_settings={**GLOBAL_CONTEXT_SETTINGS}, help="Calculating test metrics for multilabel taggers.")
@click.option('-v', '--version', is_flag=True,
              callback=partial(print_version, 'animetimm.multilabel.test'), expose_value=False, is_eager=True)
@click.option('--num-workers', '-nw', default=32, type=int, help='Number of workers', show_default=True)
@click.option('--batch-size', '-bs', default=32, type=int, help='Batch size', show_default=True)
@click.option('--workdir', '-w', default=None, type=str, help='Workdir to save training data', show_default=True)
@click.option('--force/--non-force', default=True, help='Force re-calculate test metrics.', show_default=True)
@click.option('--need-metrics/--no-metrics', default=True, help='Need metrics to get tested.', show_default=True)
@click.option('--visibility', '-V', default='manual', type=click.Choice(['private', 'public', 'gated', 'manual']),
              help='Visibility when creating model repository (will be ignored when model repository already exist.',
              show_default=True)
@click.option('--repository', '-r', default=None, help='Repository for uploading model', show_default=True)
@click.option('--tag', '-t', 'tags', multiple=True, type=str, help='Append tags for repository name', show_default=True)
@click.option('--title', '-T', default=None, type=str, help='Title for repository', show_default=True)
@click.option('--description', '-desc', default=None, type=str, help='Description for repository', show_default=True)
@click.option('--use-test-size/--use-eval-size', 'use_test_size', default=True, help='Use test size for inference',
              show_default=True)
@click.option('-l', '--licence', '--license', 'license', type=click.Choice(VALID_LICENCES), default='mit',
              help='Licence for repository.', show_default=True)
def cli(workdir, num_workers, batch_size, force, need_metrics, repository, visibility, tags, title, description,
        use_test_size, license):
    logging.try_init_root(logging.INFO)
    if need_metrics:
        test(
            workdir=workdir,
            num_workers=num_workers,
            batch_size=batch_size,
            force=force,
            use_test_size=use_test_size,
        )

    export(
        workdir=workdir,
        repo_id=repository,
        visibility=visibility,
        logfile_anonymous=True,
        append_tags=tags,
        title=title,
        description=description,
        license=license,
    )


if __name__ == '__main__':
    cli()
