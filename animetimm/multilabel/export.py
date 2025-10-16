import datetime
import glob
import json
import os
import re
import shutil
from functools import partial
from pprint import pformat
from tempfile import TemporaryDirectory
from textwrap import indent
from typing import Optional, Literal, List

import click
import numpy as np
import pandas as pd
import torch
from PIL import Image
from accelerate import Accelerator
from datasets import load_dataset
from ditk import logging
from hbutils.encoding import sha3
from hbutils.string import titleize
from hbutils.testing import vpip
from hfutils.operate import get_hf_client, upload_directory_as_directory
from hfutils.repository import hf_hub_repo_url, hf_hub_repo_file_url
from hfutils.utils import hf_normpath
from huggingface_hub import hf_hub_url
from imgutils.generic import MultiLabelTIMMModel as _OriginMultiLabelTIMMModel
from imgutils.preprocess import create_pillow_transforms
from imgutils.preprocess.torchvision import PadToSize, parse_torchvision_transforms, create_torchvision_transforms
from imgutils.utils import open_onnx_model
from thop import clever_format
from timm.models._hub import save_for_hf
from torchvision.transforms import Compose

from .augmentation import create_transforms
from .dataset import _get_normalize_from_repo_id
from .test import test
from ..dataset import load_pretrained_tag
from ..model import Model
from ..onnx import export_model_to_onnx, ExportedONNXNotUniqueError
from ..utils import GLOBAL_CONTEXT_SETTINGS, print_version, is_tensorboard_has_content, \
    VALID_LICENCES, torch_model_profile_via_calflops

_LOG_FILE_PATTERN = re.compile(r'^events\.out\.tfevents\.(?P<timestamp>\d+)\.(?P<machine>[^.]+)\.(?P<extra>[\s\S]+)$')


class MultiLabelTIMMModel(_OriginMultiLabelTIMMModel):
    def __init__(self, repo_id: str, upload_dir: str):
        self._upload_dir = upload_dir
        _OriginMultiLabelTIMMModel.__init__(self, repo_id=repo_id)

    def _open_model(self):
        with self._lock:
            if self._model is None:
                self._model = open_onnx_model(os.path.join(self._upload_dir, 'model.onnx'))

        return self._model

    def _open_tags(self):
        with self._lock:
            if self._df_tags is None:
                self._df_tags = pd.read_csv(os.path.join(self._upload_dir, 'selected_tags.csv'), keep_default_na=False)
                with open(os.path.join(self._upload_dir, 'categories.json'), 'r') as f:
                    d_category_names = {cate_item['category']: cate_item['name'] for cate_item in json.load(f)}
                    self._name_to_categories = {}
                    for category in sorted(set(self._df_tags['category'])):
                        self._category_names[category] = d_category_names[category]
                        self._name_to_categories[self._category_names[category]] = category

        return self._df_tags

    def _open_preprocess(self):
        with self._lock:
            if self._preprocess is None:
                with open(os.path.join(self._upload_dir, 'preprocess.json'), 'r') as f:
                    data_ = json.load(f)
                    test_trans = create_pillow_transforms(data_['test'])
                    val_trans = create_pillow_transforms(data_['val'])
                    self._preprocess = val_trans, test_trans

        return self._preprocess

    def _open_default_category_thresholds(self):
        with self._lock:
            if self._default_category_thresholds is None:
                try:
                    df_category_thresholds = pd.read_csv(os.path.join(self._upload_dir, 'thresholds.csv'),
                                                         keep_default_na=False)
                except (FileNotFoundError,):
                    self._default_category_thresholds = {}
                else:
                    self._default_category_thresholds = {}
                    for item in df_category_thresholds.to_dict('records'):
                        if item['category'] not in self._default_category_thresholds:
                            self._default_category_thresholds[item['category']] = item['threshold']

        return self._default_category_thresholds


def _name_safe(name_text):
    return re.sub(r'[\W_]+', '_', name_text).strip('_')


def export(workdir: str, repo_id: Optional[str] = None,
           visibility: Literal['private', 'public', 'gated', 'manual'] = 'private',
           logfile_anonymous: bool = True, append_tags: Optional[List[str]] = None,
           title: Optional[str] = None, description: Optional[str] = None, license: str = 'mit',
           onnx_opset_version: int = 14, no_onnx_export: bool = False, namespace: str = 'animetimm'):
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
        meta_info['type'] = 'multilabel'

        dataset_repo_id = meta_info['train']['dataset']
        use_normalize = meta_info['train'].get('use_normalize', False) or False
        checkpoints = os.path.join(workdir, 'checkpoints')
        best_ckpt_zip_file = os.path.join(checkpoints, 'best.zip')
        logging.info(f'Loading model from {best_ckpt_zip_file!r} ...')
        model, meta, metrics = Model.load_from_zip(best_ckpt_zip_file)
        if not use_test_size:
            model.pretrained_cfg['test_input_size'] = model.pretrained_cfg['input_size']
        model.pretrained_cfg['license'] = license
        model.module.pretrained_cfg.update(model.pretrained_cfg)
        model.module.eval()

        model: Model
        pretrained_tag = meta_info['train'].get('pretrained_tag') or load_pretrained_tag(dataset_repo_id)
        logging.info(f'Pretrained tag {pretrained_tag!r} found for dataset {dataset_repo_id!r}.')
        model.pretrained_tag = pretrained_tag

        model_name = '.'.join([model.architecture, pretrained_tag, *append_tags])
        repo_id = repo_id or f'{namespace}/{model_name}'
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
                'tags': model.tags,
            },
            model_args=model.get_actual_model_args(),
            safe_serialization='both',
        )

        print(metrics)
        metrics_file = os.path.join(upload_dir, 'metrics.json')
        category_thresholds = None
        with open(metrics_file, 'w') as f:
            metrics_info = {}
            if os.path.exists(os.path.join(workdir, 'test_metrics.json')):
                with open(os.path.join(workdir, 'test_metrics.json'), 'r') as mf:
                    ji = json.load(mf)
                    category_thresholds = ji.pop('categories', None)
                    metrics_info['test'] = ji
            metrics_info['val'] = {
                key.split('/', maxsplit=1)[-1]: value
                for key, value in metrics.items()
                if isinstance(value, (type(None), int, float, str))
            }
            logging.info(f'Writing metrics to {metrics_file!r}:\n{pformat(metrics_info)}')
            json.dump(metrics_info, f, sort_keys=True, ensure_ascii=False, indent=4)

        df_eval_tags: pd.DataFrame = metrics['df_details']
        df_tags = df_eval_tags.rename(columns={'mcc': 'val_mcc', 'f1': 'val_f1',
                                               'precision': 'val_precision', 'recall': 'val_recall'})
        if os.path.exists(os.path.join(workdir, 'test_tags.csv')):
            df_test_tags = pd.read_csv(os.path.join(workdir, 'test_tags.csv'), keep_default_na=False)
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
        if use_normalize:
            mean, std = _get_normalize_from_repo_id(repo_id=dataset_repo_id)
        else:
            mean, std = None, None
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
                mean=mean,
                std=std,
            )
            logging.info(f'Eval transform:\n{eval_trans}')

            test_trans, _ = create_transforms(
                timm_model=model.module,
                is_training=False,
                use_test_size=use_test_size,
                noise_level=0,
                rotation_ratio=0,
                mixup_alpha=0.0,
                cutout_patches=0,
                cutout_max_pct=0.0,
                random_resize_method=False,
                pre_align=meta_info['train']['pre_align'],
                align_size=meta_info['train']['align_size'],
                mean=mean,
                std=std,
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
                    wrap_mode='sigmoid',
                    opset_version=onnx_opset_version,
                    verbose=False,
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

        with open(hf_client.hf_hub_download(
                repo_id=dataset_repo_id,
                repo_type='dataset',
                filename='categories.json',
        ), 'r') as f:
            d_category_names = {cate_item['category']: cate_item['name'] for cate_item in json.load(f)}

        infer_model = MultiLabelTIMMModel(repo_id=repo_id, upload_dir=upload_dir)
        image_key = meta_info['train'].get('image_key', 'webp')
        dataset = load_dataset(dataset_repo_id, split='test')

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

            title = title or f'Anime Tagger {model_name}'
            print(f'# {title}', file=f)
            print(f'', file=f)
            if description:
                print(f'{description}', file=f)
                print(f'', file=f)

            s_flops, s_params, s_macs = clever_format([flops, params, macs], "%.1f")
            print(f'## Model Details', file=f)
            print(f'', file=f)
            print(f'- **Model Type:** Multilabel Image classification / feature backbone', file=f)
            print(f'- **Model Stats:**', file=f)
            print(f'  - Params: {s_params}', file=f)
            print(f'  - FLOPs / MACs: {s_flops} / {s_macs}', file=f)
            print(f'  - Image size: train = {dummy_input_val.shape[-1]} x {dummy_input_val.shape[-2]}, '
                  f'test = {dummy_input_test.shape[-1]} x {dummy_input_test.shape[-2]}', file=f)
            print(f'- **Dataset:** [{dataset_repo_id}]'
                  f'({hf_hub_repo_url(repo_id=dataset_repo_id, repo_type="dataset", endpoint="https://huggingface.co")})',
                  file=f)
            print(f'  - Tags Count: {len(df_tags)}', file=f)
            for category in sorted(set(df_tags['category'])):
                print(f'    - {titleize(d_category_names[category])} (#{category}) Tags Count: '
                      f'{len(df_tags[df_tags["category"] == category])}', file=f)
            print(f'', file=f)

            print(f'## Results', file=f)
            print(f'', file=f)
            eval_threshold = meta_info['train'].get('eval_threshold', 0.4)
            s_records = [{
                '#': 'Validation',
                f'Macro@{eval_threshold:.2f} (F1/MCC/P/R)': '%.3f / %.3f / %.3f / %.3f' % (
                    metrics_info['val']['macro_f1'],
                    metrics_info['val']['macro_mcc'],
                    metrics_info['val']['macro_precision'],
                    metrics_info['val']['macro_recall']
                ),
                f'Micro@{eval_threshold:.2f} (F1/MCC/P/R)': '%.3f / %.3f / %.3f / %.3f' % (
                    metrics_info['val']['micro_f1'],
                    metrics_info['val']['micro_mcc'],
                    metrics_info['val']['micro_precision'],
                    metrics_info['val']['micro_recall'],
                ),
            }]
            if 'test' in metrics_info:
                test_threshold = metrics_info.get('test_threshold', 0.4)
                s_records.append({
                    '#': 'Test',
                    f'Macro@{test_threshold:.2f} (F1/MCC/P/R)': '%.3f / %.3f / %.3f / %.3f' % (
                        metrics_info['test']['macro_f1'],
                        metrics_info['test']['macro_mcc'],
                        metrics_info['test']['macro_precision'],
                        metrics_info['test']['macro_recall'],
                    ),
                    f'Micro@{test_threshold:.2f} (F1/MCC/P/R)': '%.3f / %.3f / %.3f / %.3f' % (
                        metrics_info['test']['micro_f1'],
                        metrics_info['test']['micro_mcc'],
                        metrics_info['test']['micro_precision'],
                        metrics_info['test']['micro_recall'],
                    ),
                    **(
                        {
                            'Macro@Best (F1/P/R)': '%.3f / %.3f / %.3f' % (
                                df_tags['best_f1'].mean(),
                                df_tags['best_precision'].mean(),
                                df_tags['best_recall'].mean(),
                            ),
                        } if os.path.exists(os.path.join(workdir, 'test_tags.csv')) else {}
                    )
                })
            else:
                test_threshold = None
            df_s = pd.DataFrame(s_records)
            df_s = df_s.replace(np.nan, '---')
            print(df_s.to_markdown(index=False, stralign='center', numalign='center'), file=f)
            print(f'', file=f)
            if test_threshold is None or abs(eval_threshold - test_threshold) < 1e-4:
                print(f'* `Macro/Micro@{eval_threshold:.2f}` means the metrics '
                      f'on the threshold {eval_threshold:.2f}.', file=f)
            else:
                print(f'* `Macro/Micro@{eval_threshold:.2f}`, `Macro/Micro@{test_threshold:.2f}` '
                      f'means the metrics on the threshold {eval_threshold:.2f} (validation) '
                      f'and {test_threshold:.2f} (test).', file=f)
            if 'test' in metrics_info and os.path.exists(os.path.join(workdir, 'test_tags.csv')):
                print('* `Macro@Best` means the mean metrics on the tag-level thresholds on each tags, '
                      'which should have the best F1 scores.', file=f)
            print(f'', file=f)

            if os.path.exists(os.path.join(workdir, 'test_tags.csv')):
                print(f'## Thresholds', file=f)
                print(f'', file=f)

                threshold_file = os.path.join(upload_dir, 'thresholds.csv')
                logging.info(f'Saving threshold file {threshold_file!r} ...')
                t_records, ts_records = [], []
                categories = []
                test_threshold = metrics_info.get('test_threshold', 0.4)
                for item in (category_thresholds or []):
                    t_records.append({
                        'category': item['category'],
                        'name': d_category_names[item['category']],
                        'alpha': item.get('alpha', 1.0),
                        'threshold': item['best_threshold'],
                        'f1': item['best_f1'],
                        'precision': item['best_precision'],
                        'recall': item['best_recall'],
                    })
                    ts_records.append({
                        'Category': item['category'],
                        'Name': d_category_names[item['category']],
                        'Alpha': '%.2f' % item.get('alpha', 1.0),
                        'Threshold': '%.3f' % item['best_threshold'],
                        'Micro@Thr (F1/P/R)': '%.3f / %.3f / %.3f' % (
                            item['best_f1'],
                            item['best_precision'],
                            item['best_recall'],
                        ),
                        f'Macro@{test_threshold:.2f} (F1/P/R)': '%.3f / %.3f / %.3f' % (
                            df_tags[df_tags['category'] == item['category']]['test_f1'].mean(),
                            df_tags[df_tags['category'] == item['category']]['test_precision'].mean(),
                            df_tags[df_tags['category'] == item['category']]['test_recall'].mean(),
                        ),
                        **(
                            {
                                'Macro@Best (F1/P/R)': '%.3f / %.3f / %.3f' % (
                                    df_tags[df_tags['category'] == item['category']]['best_f1'].mean(),
                                    df_tags[df_tags['category'] == item['category']]['best_precision'].mean(),
                                    df_tags[df_tags['category'] == item['category']]['best_recall'].mean(),
                                ),
                            } if os.path.exists(os.path.join(workdir, 'test_tags.csv')) else {}
                        )
                    })
                    categories.append({
                        'category': item['category'],
                        'name': d_category_names[item['category']],
                    })

                pd.DataFrame(t_records).to_csv(threshold_file, index=False)
                print(pd.DataFrame(ts_records).to_markdown(index=False, stralign='center', numalign='center'), file=f)
                print(f'', file=f)
                print(f'* `Micro@Thr` means the metrics on the category-level suggested thresholds, '
                      f'which are listed in the table above.', file=f)
                print(f'* `Macro@{test_threshold:.2f}` means the metrics '
                      f'on the threshold {test_threshold:.2f}.', file=f)
                if os.path.exists(os.path.join(workdir, 'test_tags.csv')):
                    print(f'* `Macro@Best` means the metrics on the tag-level thresholds on each tags, '
                          'which should have the best F1 scores.', file=f)
                print(f'', file=f)
                print(f'For tag-level thresholds, you can find them in [selected_tags.csv]'
                      f'({hf_hub_url(repo_id=repo_id, repo_type="model", filename="selected_tags.csv", endpoint="https://huggingface.co")}).',
                      file=f)
                print(f'', file=f)

            else:
                categories = []
                for category in sorted(set(df_tags['category'].tolist())):
                    categories.append({
                        'category': category,
                        'name': d_category_names[category],
                    })

            categories_file = os.path.join(upload_dir, 'categories.json')
            with open(categories_file, 'w') as cf:
                json.dump(categories, cf, sort_keys=True, indent=4, ensure_ascii=False)

            print(f'## How to Use', file=f)
            print(f'', file=f)

            imgutils_version = str(vpip('dghs-imgutils')._actual_version)
            sample_input = dataset[0][image_key]
            sample_input_file = os.path.join(upload_dir, 'sample.webp')
            sample_input_relfile = hf_normpath(os.path.relpath(sample_input_file, upload_dir))
            sample_input.save(sample_input_file)
            sample_input_url = hf_hub_url(repo_id=repo_id, repo_type='model', filename=sample_input_relfile)
            sample_input_page_url = hf_hub_repo_file_url(repo_id=repo_id, repo_type='model', path=sample_input_relfile)

            print(f'We provided a sample image for our code samples, '
                  f'you can find it [here]({sample_input_page_url}).', file=f)
            print(f'', file=f)

            print(f'### Use TIMM And Torch', file=f)
            print(f'', file=f)
            print(f'Install [dghs-imgutils](https://github.com/deepghs/imgutils), '
                  f'[timm](https://github.com/huggingface/pytorch-image-models) '
                  f'and other necessary requirements with the following command', file=f)
            print(f'', file=f)
            print(f'```shell', file=f)
            print(f'pip install \'dghs-imgutils>={imgutils_version}\' torch huggingface_hub timm pillow pandas', file=f)
            print(f'```', file=f)
            print(f'', file=f)
            print(f'After that you can load this model with timm library, and use it for train, validation and test, '
                  f'with the following code', file=f)
            print(f'', file=f)
            print(f'```python', file=f)
            print(f'import json', file=f)
            print(f'', file=f)
            print(f'import pandas as pd', file=f)
            print(f'import torch', file=f)
            print(f'from huggingface_hub import hf_hub_download', file=f)
            print(f'from imgutils.data import load_image', file=f)
            print(f'from imgutils.preprocess import create_torchvision_transforms', file=f)
            print(f'from timm import create_model', file=f)
            print(f'', file=f)
            print(f"repo_id = {repo_id!r}", file=f)
            print(f"model = create_model(f'hf-hub:{{repo_id}}', pretrained=True)", file=f)
            print(f'model.eval()', file=f)
            print(f'', file=f)
            print(
                f"with open(hf_hub_download(repo_id=repo_id, repo_type='model', filename='preprocess.json'), 'r') as f:",
                file=f)
            print(f"    preprocessor = create_torchvision_transforms(json.load(f)['test'])", file=f)
            tv_preprocess = create_torchvision_transforms(trans['test'])
            print(indent(str(tv_preprocess), prefix="# "), file=f)
            print(f'', file=f)
            print(f"image = load_image({sample_input_url!r})", file=f)
            input_ = tv_preprocess(sample_input).unsqueeze(0)
            model, _, _ = Model.load_from_zip(best_ckpt_zip_file)
            model = model.module
            model.eval()
            with torch.no_grad():
                output = model(input_)
                prediction = torch.sigmoid(output)[0]
            print(f'input_ = preprocessor(image).unsqueeze(0)', file=f)
            print(f'# input_, shape: {input_.shape!r}, dtype: {input_.dtype!r}', file=f)
            print(f'with torch.no_grad():', file=f)
            print(f'    output = model(input_)', file=f)
            print(f'    prediction = torch.sigmoid(output)[0]', file=f)
            print(f'# output, shape: {output.shape!r}, dtype: {output.dtype!r}', file=f)
            print(f'# prediction, shape: {prediction.shape!r}, dtype: {prediction.dtype!r}', file=f)
            print(f'', file=f)
            print(f'df_tags = pd.read_csv(', file=f)
            print(f"    hf_hub_download(repo_id=repo_id, repo_type='model', filename='selected_tags.csv'),", file=f)
            print(f'    keep_default_na=False', file=f)
            print(f')', file=f)
            print(f"tags = df_tags['name']", file=f)
            if 'best_threshold' in df_tags:
                print(f"mask = prediction.numpy() >= df_tags['best_threshold']", file=f)
            else:
                print(f"mask = prediction.numpy() >= 0.4", file=f)
            print(f'print(dict(zip(tags[mask].tolist(), prediction[mask].tolist())))', file=f)

            tags = df_tags['name']
            if 'best_threshold' in df_tags:
                mask = prediction.numpy() >= df_tags['best_threshold']
            else:
                mask = prediction.numpy() >= 0.4
            dt = dict(zip(tags[mask].tolist(), prediction[mask].tolist()))
            print(f'{indent(pformat(dt, sort_dicts=False), prefix="# ")}', file=f)
            print(f'```', file=f)
            print(f'', file=f)

            if not no_onnx_export:
                print(f'### Use ONNX Model For Inference', file=f)
                print(f'', file=f)
                print(f'Install [dghs-imgutils](https://github.com/deepghs/imgutils) with the following command',
                      file=f)
                print(f'', file=f)
                print(f'```shell', file=f)
                print(f'pip install \'dghs-imgutils>={imgutils_version}\'', file=f)
                print(f'```', file=f)
                print(f'', file=f)
                print(f'Use `multilabel_timm_predict` function with the following code', file=f)
                print(f'', file=f)
                print(f'```python', file=f)
                print(f'from imgutils.generic import multilabel_timm_predict', file=f)
                print(f'', file=f)

                default_fmt = tuple(
                    d_category_names[category] for category in sorted(set(df_tags['category'].tolist())))
                var_names = tuple(map(_name_safe, default_fmt))
                print(f'{", ".join(var_names)} = multilabel_timm_predict(', file=f)
                print(f'    {sample_input_url!r},', file=f)
                print(f'    repo_id={repo_id!r},', file=f)
                print(f'    fmt={(default_fmt if len(default_fmt) != 1 else default_fmt[0])!r},', file=f)
                print(f')', file=f)
                print(f'', file=f)
                values = infer_model.predict(
                    sample_input,
                    fmt=default_fmt
                )
                for rt_name, rt_val in zip(var_names, values):
                    print(f'print({rt_name})', file=f)
                    print(f'{indent(pformat(rt_val, sort_dicts=False), prefix="# ")}', file=f)
                print(f'```', file=f)
                print(f'', file=f)
                print('For further information, see [documentation of function multilabel_timm_predict]'
                      '(https://dghs-imgutils.deepghs.org/main/api_doc/generic/multilabel_timm.html#multilabel-timm-predict).',
                      file=f)
                print(f'', file=f)

            print(f'## Citation', file=f)
            print(f'', file=f)
            print(f'```', file=f)
            print(f'@misc{{{_name_safe(repo_id.split("/")[-1])},', file=f)
            print(f'  title        = {{{title}}},', file=f)
            print(f'  author       = {{narugo1992 and Deep Generative anime Hobbyist Syndicate (DeepGHS)}},', file=f)
            print(f'  year         = {{{datetime.datetime.now().year}}},', file=f)
            print(f'  howpublished = {{\\url{{{hf_hub_repo_url(repo_id=repo_id, repo_type="model")}}}}},',
                  file=f)
            print(f'  note         = {{A large-scale anime-style image classification model '
                  f'based on {model.architecture} architecture '
                  f'for multi-label tagging with {len(df_tags)} tags, trained on anime dataset {pretrained_tag} '
                  f'(\\url{{{hf_hub_repo_url(repo_id=dataset_repo_id, repo_type="dataset")}}}). '
                  f'Model parameters: {s_params}, FLOPs: {s_flops}, '
                  f'input resolution: {dummy_input_test.shape[-1]}×{dummy_input_test.shape[-2]}.}},',
                  file=f)
            print(f'  license      = {{{license}}}', file=f)
            print(f'}}', file=f)
            print(f'```', file=f)
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
@click.option('--test-threshold', '-tt', default=0.4, type=float, help='Test threshold', show_default=True)
@click.option('--tag-categories', '-tc', multiple=True, type=int, help='Tag categories (multiple)', show_default=True)
@click.option('--seen-tag-keys', '-stk', multiple=True, help='Seen tag keys (multiple)', show_default=True)
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
@click.option('-opv', '--onnx-opset-version', 'onnx_opset_version', default=14, type=int,
              help='OpSet Version of ONNX Export.', show_default=True)
@click.option('--no-onnx-export', 'no_onnx_export', is_flag=True, default=False, type=bool,
              help='No ONNX model to export, just save the weights.', show_default=True)
@click.option('-ns', '--namespace', 'namespace', default='animetimm', type=str, show_default=True,
              help='Namespace for the publish repository')
def cli(workdir, num_workers, batch_size, test_threshold, tag_categories, seen_tag_keys, force, need_metrics,
        repository, visibility, tags, title, description, use_test_size, license, onnx_opset_version,
        no_onnx_export, namespace):
    logging.try_init_root(logging.INFO)
    accelerator = Accelerator(
        # mixed_precision=self.cfgs.mixed_precision,
        step_scheduler_with_optimizer=False,
    )

    if need_metrics:
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
            accelerator=accelerator,
        )

    if accelerator.is_main_process:
        export(
            workdir=workdir,
            repo_id=repository,
            visibility=visibility,
            logfile_anonymous=True,
            append_tags=tags,
            title=title,
            description=description,
            license=license,
            onnx_opset_version=onnx_opset_version,
            no_onnx_export=no_onnx_export,
            namespace=namespace,
        )


if __name__ == '__main__':
    cli()
