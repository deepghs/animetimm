import json
import os
import random
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
         name_filter: Optional[Callable[[str, ], bool]] = None, max_cnt_per_level: int = 10):
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()
    delete_detached_cache()
    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)
        attr_lines = hf_fs.read_text(f'{repository}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(f'{repository}/.gitattributes', os.linesep.join(attr_lines))

    if not drop_previous and hf_client.file_exists(
            repo_id=repository,
            repo_type='model',
            filename='models.parquet',
    ):
        df_models = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='model',
            filename='models.parquet',
        ))
        d_models = {item['name']: item for item in df_models.to_dict('records')}
    else:
        d_models = {}

    all_pretrained = list_pretrained()[:100]
    random.shuffle(all_pretrained)
    for model_name in tqdm(all_pretrained, desc='Scanning Models'):
        if model_name.startswith('tf_'):
            continue
        if name_filter and not name_filter(model_name):
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
            'level': level_id,
            'level_name': level_name,
            'num_classes': num_classes,
            'num_features': num_features,
            'downloads': repo_info.downloads,
            'downloads_all_time': repo_info.downloads_all_time,
            'likes': repo_info.likes,
        }
        d_models[row['name']] = row

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

                df_models_level = df_models_level.sort_values(by=['downloads', 'likes', 'downloads_all_time'],
                                                              ascending=False)
                d_arch_models = {}
                for item in df_models_level.to_dict('records'):
                    level_name = item['level_name']
                    if item['architecture'] not in d_arch_models:
                        d_arch_models[item['architecture']] = {
                            'Name': f'[{item["model_name"]}]({hf_hub_repo_url(repo_id=item["repo_id"], repo_type="model")})',
                            'Architecture': item['architecture'],
                            'Params': clever_format(item['params'], '%.1f'),
                            'Num Classes': item['num_classes'],
                            'Num Features': item['num_features'],
                            'Downloads (Current / All Time)': f'{item["downloads"]} / {item["downloads_all_time"]}',
                            'Likes': item['likes'],
                        }
                    if len(d_arch_models) >= max_cnt_per_level:
                        break

                df_level = pd.DataFrame(list(d_arch_models.values()))
                df_level = df_level.sort_values(by=['downloads', 'likes', 'downloads_all_time'], ascending=False)

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


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    sync()
