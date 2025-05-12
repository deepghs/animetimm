import json
import os
from tempfile import TemporaryDirectory

import pandas as pd
from ditk import logging
from timm.models._hub import save_for_hf

from .dataset import load_pretrained_tag
from ..model import Model


def export(workdir: str):
    with TemporaryDirectory() as upload_dir:
        with open(os.path.join(workdir, 'meta.json'), 'r') as f:
            meta_info = json.load(f)

        dataset_repo_id = meta_info['train']['dataset']
        checkpoints = os.path.join(workdir, 'checkpoints')
        best_ckpt_zip_file = os.path.join(checkpoints, 'best.zip')
        model, meta, metrics = Model.load_from_zip(best_ckpt_zip_file)

        model: Model
        pretrained_tag = meta_info['train'].get('pretrained_tag') or load_pretrained_tag(dataset_repo_id)
        logging.info(f'Pretrained tag {pretrained_tag!r} found for dataset {dataset_repo_id!r}.')
        model.pretrained_tag = pretrained_tag

        save_for_hf(
            model.module,
            upload_dir,
            model_config={
                'tags': model.tags,
            },
            model_args=model.model_args,
            safe_serialization='both',
        )

        with open(os.path.join(upload_dir, 'metrics.json'), 'w') as f:
            metrics_info = {}
            if os.path.exists(os.path.join(workdir, 'test_metrics.json')):
                with open(os.path.join(workdir, 'test_metrics.json'), 'r') as mf:
                    metrics_info['test'] = json.load(mf)
            metrics_info['val'] = {
                key: value
                for key, value in metrics.items()
                if isinstance(value, (type(None), int, float, str))
            }
            json.dump(metrics_info, f, sort_keys=True, ensure_ascii=False, indent=4)

        df_eval_tags: pd.DataFrame = metrics['details']
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
        df_tags.to_csv(os.path.join(upload_dir, 'selected_tags.csv'), index=False)

        os.system(f'tree {upload_dir!r}')


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    W = os.environ['W']
    export(W)
