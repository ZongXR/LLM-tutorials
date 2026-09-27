# -*- coding: utf-8 -*-
"""
手动batch推理
"""
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, PeftModel, LoraConfig


if __name__ == '__main__':
    prompt = "请判断以下文字的感情色彩\n消极：0、中性：1、积极：2\n"
    id2label = {
        0: "消极",
        1: "中性",
        2: "积极"
    }
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        num_labels=3,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()}
    )
    model = PeftModel.from_pretrained(model, "adapter")
    tokenizer = AutoTokenizer.from_pretrained("adapter")
    dataset = load_dataset("csv", data_files={
        "train": "data/train.csv",
        "test": "data/test.csv"
    }, split="test")
    batch_size = 16

    model.config.pad_token_id = tokenizer.pad_token_id
    dataset = dataset.map(lambda x: tokenizer([prompt + xx for xx in x["text"]], padding=True, truncation=True), batched=True)
    model.eval()
    for i in range(0, len(dataset), batch_size):
        # inputs = {k: torch.tensor(v).to("cuda:0") for k, v in dataset[i : i + batch_size].items() if k in ("input_ids", "attention_mask")}
        inputs = dataset.select(range(i, i + batch_size)).with_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            device="cuda:0"
        )[::]
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = logits.argmax(dim=-1).cpu().numpy().tolist()
        print([model.config.id2label[x] for x in preds])
        break
