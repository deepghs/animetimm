import json
import os
import random
import time
from typing import Optional, Callable

import pandas as pd
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.cache import delete_detached_cache
from hfutils.operate import get_hf_client, get_hf_fs, upload_directory_as_directory
from hfutils.repository import hf_hub_repo_url
from huggingface_hub import hf_hub_download
from thop import clever_format
from timm import list_pretrained
from tqdm import tqdm

from .level import classify_model_by_params


def sync(repository: str = 'deepghs/timms_index', drop_previous: bool = False,
         name_filter: Optional[Callable[[str, ], bool]] = None, max_cnt_per_level: int = 10,
         deploy_span: float = 5 * 60):
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()
    delete_detached_cache()
    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)
        attr_lines = hf_fs.read_text(f'datasets/{repository}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(f'datasets/{repository}/.gitattributes', os.linesep.join(attr_lines))

    df_imagenet = pd.read_csv(hf_client.hf_hub_download(
        repo_id='deepghs/timm_results',
        repo_type='dataset',
        filename='results-imagenet.csv',
    ))
    d_imagenet = {item['model']: item for item in df_imagenet.to_dict('records')}

    if not drop_previous and hf_client.file_exists(
            repo_id=repository,
            repo_type='dataset',
            filename='models.parquet',
    ):
        df_models = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='dataset',
            filename='models.parquet',
        ))
        d_models = {item['name']: item for item in df_models.to_dict('records')}
    else:
        d_models = {}

    _last_update, has_update = None, False
    _total_count = len(d_models)

    def _deploy(force=False):
        nonlocal _last_update, has_update, _total_count

        if not has_update:
            return
        if not force and _last_update is not None and _last_update + deploy_span > time.time():
            return

        with TemporaryDirectory() as upload_dir:
            df_models = pd.DataFrame(list(d_models.values()))
            df_models.to_parquet(os.path.join(upload_dir, 'models.parquet'), index=False)

            with open(os.path.join(upload_dir, 'README.md'), 'w') as f:
                print(f'---', file=f)
                print(f'tags:', file=f)
                print(f'- image-classification', file=f)
                print(f'- timm', file=f)
                print(f'- animetimm', file=f)
                print(f'- dghs-imgutils', file=f)
                print(f'---', file=f)
                print(f'', file=f)

                print(f'Index for timm models of different size levels', file=f)
                print(f'', file=f)

                print(f'{plural_word(len(df_models), "model")} in total.', file=f)
                print(f'', file=f)
                print(f'For each level, only the top-{max_cnt_per_level} '
                      f'most popular pretrained weight of each architecture will be listed here.', file=f)
                print(f'', file=f)

                for level_id in range(11):
                    level_name = None
                    df_models_level = df_models[df_models['level'] == level_id]
                    if len(df_models_level) == 0:
                        continue

                    df_models_level = df_models_level.sort_values(by=['top5', 'top1'], ascending=False)
                    arch_models = []
                    exist_archs = set()
                    for item in df_models_level.to_dict('records'):
                        level_name = item['level_name']
                        if item['architecture'] not in exist_archs:
                            arch_models.append({
                                'Name': f'[{item["name"]}]({hf_hub_repo_url(repo_id=item["repo_id"], repo_type="model")})',
                                'Architecture': item['architecture'],
                                'Params': clever_format(item['params'], '%.1f'),
                                'Input Size': item['img_size'],
                                'Top-1': f'{item["top1"]:.2f}%',
                                'Top-5': f'{item["top5"]:.2f}%',
                                'Num Classes': item['num_classes'],
                                'Num Features': item['num_features'],
                                'Downloads': item["downloads"],
                                'Likes': item['likes'],
                            })
                            exist_archs.add(item['architecture'])
                        if len(arch_models) >= max_cnt_per_level:
                            break

                    df_level = pd.DataFrame(arch_models)

                    print(f'## {level_id} - {level_name}', file=f)
                    print(f'', file=f)
                    print(df_level.to_markdown(index=False), file=f)
                    print(f'', file=f)

            upload_directory_as_directory(
                repo_id=repository,
                repo_type='dataset',
                local_directory=upload_dir,
                path_in_repo='.',
                message=f'Sync for {plural_word(len(df_models), "model")}',
            )
            has_update = False
            _total_count = len(df_models)
            _last_update = time.time()

    all_pretrained = list_pretrained()
    random.shuffle(all_pretrained)
    for model_name in tqdm(all_pretrained, desc='Scanning Models'):
        if model_name.startswith('tf_'):
            continue
        if name_filter and not name_filter(model_name):
            continue
        if model_name not in d_imagenet:
            continue
        if model_name in d_models:
            logging.info(f'Model {model_name!r} already exist, skipped.')
            continue

        model_repo_id = f'timm/{model_name}'
        if not hf_client.repo_exists(repo_id=model_repo_id, repo_type='model'):
            logging.warn(f'Repo {model_repo_id!r} not exist, skipped.')
            continue

        logging.info(f'Checking {model_repo_id!r} ...')
        repo_info = hf_client.repo_info(repo_id=model_repo_id, repo_type='model')
        if repo_info.safetensors and repo_info.safetensors.total:
            params = repo_info.safetensors.total
        elif hf_client.file_exists(
                repo_id=model_repo_id,
                repo_type='model',
                filename='model.safetensors',
        ):
            params = hf_fs.size(f'{model_repo_id}/model.safetensors') // 4
        elif hf_client.file_exists(
                repo_id=model_repo_id,
                repo_type='model',
                filename='pytorch_model.bin',
        ):
            params = hf_fs.size(f'{model_repo_id}/pytorch_model.bin') // 4
        else:
            logging.warn(f'No size or model file found for {model_repo_id!r}, skipped.')
            continue

        s_params = clever_format(params, "%.1f")
        level_id, level_name = classify_model_by_params(params / 1e6)
        logging.info(f'Pretrained model {model_name!r}, params: {s_params}, level #{level_id}, name: {level_name!r}')

        with open(hf_hub_download(repo_id=model_repo_id, repo_type='model', filename='config.json'), 'r') as f:
            meta = json.load(f)
            architecture = meta['architecture']
            num_classes = meta['num_classes']
            num_features = meta['num_features']

        row = {
            'name': model_name,
            'repo_id': model_repo_id,
            'architecture': architecture,
            'params': params,
            'img_size': d_imagenet[model_name]['img_size'],
            'top1': d_imagenet[model_name]['top1'],
            'top5': d_imagenet[model_name]['top5'],
            'crop_pct': d_imagenet[model_name]['crop_pct'],
            'interpolation': d_imagenet[model_name]['interpolation'],
            'level': level_id,
            'level_name': level_name,
            'num_classes': num_classes,
            'num_features': num_features,
            'downloads': repo_info.downloads,
            'likes': repo_info.likes,
        }
        d_models[row['name']] = row
        has_update = True
        _deploy(force=False)

    has_update = True
    _deploy(force=True)


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    sync(
        max_cnt_per_level=30,
        drop_previous=False,
    )
