# -*- coding: utf-8 -*-
"""
download model from modelscope
"""
from modelscope.hub.snapshot_download import snapshot_download


if __name__ == '__main__':
    model = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    model_dir = snapshot_download(
        model,
        local_dir="./" + model,
        revision=None,  # 默认下载最新版本
        cache_dir=None,  # 默认缓存路径
    )
    print(f"模型已下载到: {model_dir}")