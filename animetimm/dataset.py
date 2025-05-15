import json

from hfutils.operate import get_hf_client


def load_pretrained_tag(repo_id: str) -> str:
    hf_client = get_hf_client()
    tag = repo_id.split('/')[-1]
    if hf_client.file_exists(
            repo_id=repo_id,
            repo_type='dataset',
            filename='pretrained_tag.json',
    ):
        with open(hf_client.hf_hub_download(
                repo_id=repo_id,
                repo_type='dataset',
                filename='pretrained_tag.json',
        ), 'r') as f:
            meta = json.load(f)

        if 'pretrained_tag' in meta:
            tag = meta['pretrained_tag']

    return tag


if __name__ == '__main__':
    print(load_pretrained_tag('animetimm/danbooru-wdtagger-v4-w640-ws-30k'))
    print(load_pretrained_tag('animetimm/danbooru-wdtagger-v4-w640-ws-50k'))
