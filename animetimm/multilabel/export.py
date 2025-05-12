import glob
import json
import os
import re
import shutil
from pprint import pformat
from tempfile import TemporaryDirectory
from typing import Optional

import pandas as pd
from PIL import Image
from ditk import logging
from hbutils.encoding import sha3
from hfutils.operate import get_hf_client
from hfutils.repository import hf_hub_repo_url
from imgutils.preprocess.torchvision import PadToSize, parse_torchvision_transforms
from thop import clever_format
from timm.models._hub import save_for_hf
from torchvision.transforms import Compose

from animetimm.utils import torch_model_profile
from .augmentation import create_transforms
from .dataset import load_pretrained_tag
from ..model import Model
from ..onnx import export_model_to_onnx

_LOG_FILE_PATTERN = re.compile(r'^events\.out\.tfevents\.(?P<timestamp>\d+)\.(?P<machine>[^.]+)\.(?P<extra>[\s\S]+)$')


def export(workdir: str, repo_id: Optional[str] = None, private: bool = False, logfile_anonymous: bool = True):
    hf_client = get_hf_client()
    with TemporaryDirectory() as upload_dir:
        meta_info_file = os.path.join(workdir, 'meta.json')
        logging.info(f'Loading meta from {meta_info_file!r} ...')
        with open(meta_info_file, 'r') as f:
            meta_info = json.load(f)

        dataset_repo_id = meta_info['train']['dataset']
        checkpoints = os.path.join(workdir, 'checkpoints')
        best_ckpt_zip_file = os.path.join(checkpoints, 'best.zip')
        logging.info(f'Loading model from {best_ckpt_zip_file!r} ...')
        model, meta, metrics = Model.load_from_zip(best_ckpt_zip_file)

        model: Model
        pretrained_tag = meta_info['train'].get('pretrained_tag') or load_pretrained_tag(dataset_repo_id)
        logging.info(f'Pretrained tag {pretrained_tag!r} found for dataset {dataset_repo_id!r}.')
        model.pretrained_tag = pretrained_tag

        repo_id = repo_id or f'animetimm/{model.architecture}.{pretrained_tag}'
        logging.info(f'Target repository: {repo_id!r}.')
        if not hf_client.repo_exists(repo_id=repo_id, repo_type='model'):
            hf_client.create_repo(repo_id=repo_id, repo_type='model', private=private)

        logging.info(f'Dumping as huggingface TIMM format to {upload_dir!r} ...')
        save_for_hf(
            model.module,
            upload_dir,
            model_config={
                'tags': model.tags,
            },
            model_args=model.model_args,
            safe_serialization='both',
        )

        print(metrics)
        metrics_file = os.path.join(upload_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            metrics_info = {}
            if os.path.exists(os.path.join(workdir, 'test_metrics.json')):
                with open(os.path.join(workdir, 'test_metrics.json'), 'r') as mf:
                    metrics_info['test'] = json.load(mf)
            metrics_info['val'] = {
                key.split('/', maxsplit=1)[-1]: value
                for key, value in metrics.items()
                if isinstance(value, (type(None), int, float, str))
            }
            logging.info(f'Writing metrics to {metrics_file!r}:\n{pformat(metrics_info)}')
            json.dump(metrics_info, f, sort_keys=True, ensure_ascii=False, indent=4)

        df_eval_tags: pd.DataFrame = metrics['df_details']
        df_tags = df_eval_tags.rename({'mcc': 'val_mcc', 'f1': 'val_f1',
                                       'precision': 'val_precision', 'recall': 'val_recall'})
        if os.path.exists(os.path.join(workdir, 'test_tags.csv')):
            df_test_tags = pd.read_csv(os.path.join(workdir, 'test_tags.csv'))
            df_tags['test_mcc'] = df_test_tags['mcc']
            df_tags['test_f1'] = df_test_tags['f1']
            df_tags['test_precision'] = df_test_tags['precision']
            df_tags['test_recall'] = df_test_tags['recall']
            df_tags['best_threshold'] = df_test_tags['best_threshold']
            df_tags['best_f1'] = df_test_tags['best_f1']
            df_tags['best_precision'] = df_test_tags['best_precision']
            df_tags['best_recall'] = df_test_tags['best_recall']
        tags_file = os.path.join(upload_dir, 'selected_tags.csv')
        logging.info(f'Dumping tags with metrics to {tags_file!r}:\n{df_tags}')
        df_tags.to_csv(tags_file, index=False)

        transforms_file = os.path.join(upload_dir, 'preprocess.json')
        logging.info(f'Dumping preprocessors to {transforms_file!r} ...')
        with open(transforms_file, 'w') as f:
            eval_trans, _ = create_transforms(
                timm_model=model.module,
                is_training=False,
                use_test_size=False,
                noise_level=0,
                rotation_ratio=0,
                mixup_alpha=0.0,
                cutout_patches=0,
                cutout_max_pct=0.0,
                random_resize_method=False,
                pre_align=meta_info['train']['pre_align'],
                align_size=meta_info['train']['align_size'],
            )
            logging.info(f'Eval transform:\n{eval_trans}')

            test_trans, _ = create_transforms(
                timm_model=model.module,
                is_training=False,
                use_test_size=True,
                noise_level=0,
                rotation_ratio=0,
                mixup_alpha=0.0,
                cutout_patches=0,
                cutout_max_pct=0.0,
                random_resize_method=False,
                pre_align=meta_info['train']['pre_align'],
                align_size=meta_info['train']['align_size'],
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

        flops, params = torch_model_profile(model=model.module, input_=dummy_input_test)
        meta_info['flops'] = flops
        meta_info['params'] = params
        new_meta_file = os.path.join(upload_dir, 'meta.json')
        logging.info(f'Saving metadata to {new_meta_file!r} ...')
        with open(new_meta_file, 'w') as f:
            json.dump(meta_info, f, indent=4, sort_keys=True, ensure_ascii=False)

        onnx_file = os.path.join(upload_dir, 'model.onnx')
        logging.info(f'Dumping to onnx file {onnx_file!r} ...')
        export_model_to_onnx(
            model=model,
            dummy_input=dummy_input_test,
            onnx_filename=onnx_file,
            metadata={**meta, 'tags': json.dumps(model.tags)},
            wrap_mode='sigmoid',
            verbose=False,
        )

        for logfile in glob.glob(os.path.join(workdir, 'events.out.tfevents.*')):
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
            print(f'---', file=f)
            print(f'- image-classification', file=f)
            print(f'- timm', file=f)
            print(f'- transformers', file=f)
            print(f'- animetimm', file=f)
            print(f'- imgutils', file=f)
            print(f'library_name: timm', file=f)
            print(f'license: gpl-3.0', file=f)
            print(f'datasets:', file=f)
            print(f'- {dataset_repo_id}', file=f)
            print(f'base_model:', file=f)
            print(f'- {model.src_repo_id}', file=f)
            print(f'---', file=f)
            print(f'', file=f)

            print(f'# Anime Tagger {model.architecture}.{pretrained_tag}', file=f)
            print(f'', file=f)

            s_flops, s_params = clever_format([flops, params], "%.1f")
            print(f'## Model Details', file=f)
            print(f'', file=f)
            print(f'- **Model Type:** Multilabel Image classification / feature backbone', file=f)
            print(f'- **Model Stats:**', file=f)
            print(f'  - Params: {s_params}', file=f)
            print(f'  - FLOPs: {s_flops}', file=f)
            print(f'  - Image size: train = {dummy_input_val.shape[-1]} x {dummy_input_val.shape[-2]}, '
                  f'test = {dummy_input_test.shape[-1]} x {dummy_input_test.shape[-2]}', file=f)
            print(f'- **Dataset:** [{dataset_repo_id}]'
                  f'({hf_hub_repo_url(repo_id=dataset_repo_id, repo_type="dataset")})', file=f)
            print(f'  - Tags Count: {len(df_tags)}', file=f)
            print(f'', file=f)

            print(f'## Results', file=f)
            print(f'', file=f)
            s_records = [{
                '#': 'Validation',
                'macro_f1': '%.3f' % metrics_info['val']['macro_f1'],
                'macro_mcc': '%.3f' % metrics_info['val']['macro_mcc'],
                'macro_precision': '%.3f' % metrics_info['val']['macro_precision'],
                'macro_recall': '%.3f' % metrics_info['val']['macro_recall'],
                'micro_f1': '%.3f' % metrics_info['val']['micro_f1'],
                'micro_mcc': '%.3f' % metrics_info['val']['micro_mcc'],
                'micro_precision': '%.3f' % metrics_info['val']['micro_precision'],
                'micro_recall': '%.3f' % metrics_info['val']['micro_recall'],
            }]
            if 'test' in metrics_info:
                s_records.append({
                    '#': 'Test',
                    'macro_f1': '%.3f' % metrics_info['test']['macro_f1'],
                    'macro_mcc': '%.3f' % metrics_info['test']['macro_mcc'],
                    'macro_precision': '%.3f' % metrics_info['test']['macro_precision'],
                    'macro_recall': '%.3f' % metrics_info['test']['macro_recall'],
                    'micro_f1': '%.3f' % metrics_info['test']['micro_f1'],
                    'micro_mcc': '%.3f' % metrics_info['test']['micro_mcc'],
                    'micro_precision': '%.3f' % metrics_info['test']['micro_precision'],
                    'micro_recall': '%.3f' % metrics_info['test']['micro_recall'],
                })
            df_s = pd.DataFrame(s_records)
            print(df_s.to_markdown(index=False), file=f)
            print(f'', file=f)

        os.system(f'tree {upload_dir!r}')


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    W = os.environ['W']
    export(
        workdir=W,
        private=True,
    )
