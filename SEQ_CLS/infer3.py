# -*- coding: utf-8 -*-
"""
使用DataLoader推理
"""
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel


if __name__ == '__main__':
    prompt = "请判断以下文字的感情色彩\n消极：0、中性：1、积极：2\n"
    id2label = {
        0: "消极",
        1: "中性",
        2: "积极"
    }
    dataset = load_dataset("csv", data_files={
        "train": "data/train.csv",
        "test": "data/test.csv"
    }, split="test")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        num_labels=3,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()}
    )
    model = PeftModel.from_pretrained(model, "adapter")
    tokenizer = AutoTokenizer.from_pretrained("adapter")
    model.config.pad_token_id = tokenizer.pad_token_id
    dataset = dataset.map(lambda x: tokenizer([prompt + xx for xx in x["text"]], padding=False, truncation=True), batched=True).select_columns(["input_ids", "attention_mask"])
    loader = DataLoader(dataset, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer), batch_size=16)
    model.eval()
    for inputs in loader:
        with torch.no_grad():
            logits = model(**{k: v.to("cuda:0") for k, v in inputs.items()}).logits
        preds = logits.argmax(dim=-1).cpu().numpy().tolist()
        print([model.config.id2label[x] for x in preds])
