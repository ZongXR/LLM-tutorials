# -*- coding: utf-8 -*-
"""
download model from hugging face mirror
"""
import os
from huggingface_hub import snapshot_download
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


if __name__ == '__main__':
    model = 'google/gemma-2b'
    local_dir = "./{path}".format(path=model)
    snapshot_download(repo_id=model,  repo_type='model', local_dir=local_dir, ignore_patterns=".gitattributes", resume_download=True)