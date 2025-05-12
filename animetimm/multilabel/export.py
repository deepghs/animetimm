import json
import os
from pprint import pprint
from tempfile import TemporaryDirectory

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

        print(model)
        pprint(('metrics', metrics))
        pprint(('meta', meta))

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
        os.system(f'cat {os.path.join(upload_dir, "config.json")!r}')


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    W = os.environ['W']
    export(W)
