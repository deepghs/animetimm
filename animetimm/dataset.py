import json

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError


def load_pretrained_tag(repo_id: str) -> str:
    tag = repo_id.split('/')[-1]
    try:
        with open(hf_hub_download(
                repo_id=repo_id,
                repo_type='dataset',
                filename='pretrained_tag.json',
        ), 'r') as f:
            meta = json.load(f)

        if 'pretrained_tag' in meta:
            tag = meta['pretrained_tag']
    except (EntryNotFoundError,):
        pass

    return tag


if __name__ == '__main__':
    print(load_pretrained_tag('animetimm/danbooru-wdtagger-v4-w640-ws-30k'))
    print(load_pretrained_tag('animetimm/danbooru-wdtagger-v4-w640-ws-50k'))
