import json
import os
from pprint import pformat
from tempfile import TemporaryDirectory

import pandas as pd
from ditk import logging
from imgutils.preprocess.torchvision import PadToSize, parse_torchvision_transforms
from timm.models._hub import save_for_hf
from torchvision.transforms import Compose

from .augmentation import create_transforms
from .dataset import load_pretrained_tag
from ..model import Model


def export(workdir: str):
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

        metrics_file = os.path.join(upload_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            metrics_info = {}
            if os.path.exists(os.path.join(workdir, 'test_metrics.json')):
                with open(os.path.join(workdir, 'test_metrics.json'), 'r') as mf:
                    metrics_info['test'] = json.load(mf)
            metrics_info['val'] = {
                key: value
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

        transforms_file = os.path.join(workdir, 'preprocess.json')
        logging.info(f'Dumping preprocessors to {transforms_file!r} ...')
        with open(transforms_file, 'w') as f:
            eval_trans = create_transforms(
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

            test_trans = create_transforms(
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
                pre_trans.append(PadToSize(size=meta_info['align_size']))
            pre_trans = Compose(pre_trans)
            logging.info(f'Pre transform:\n{pre_trans}')

            trans = {
                'val': parse_torchvision_transforms(eval_trans),
                'test': parse_torchvision_transforms(test_trans),
                'pre': parse_torchvision_transforms(pre_trans),
            }
            json.dump(trans, f, ensure_ascii=False, sort_keys=True, indent=4)

        os.system(f'tree {upload_dir!r}')


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    W = os.environ['W']
    export(W)
