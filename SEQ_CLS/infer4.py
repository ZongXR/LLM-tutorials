# -*- coding: utf-8 -*-
"""
直接使用Trainer推理
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, PeftModel
from datasets import load_dataset


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
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        num_labels=3,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()}
    )
    model = PeftModel.from_pretrained(model, "adapter")
    tokenizer = AutoTokenizer.from_pretrained("adapter")
    model.config.pad_token_id = tokenizer.pad_token_id
    dataset = dataset.map(lambda x: tokenizer([prompt + xx for xx in x["text"]], padding=True, truncation=True), batched=True).rename_column("label", "labels")
    train_args = TrainingArguments(
        output_dir="./checkpoints",  # directory to save and repository id
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=8,
        seed=42,
        data_seed=42,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        processing_class=tokenizer
    )
    model.eval()
    logits = trainer.predict(dataset.select_columns(["input_ids", "attention_mask"])).predictions
    preds = logits.argmax(axis=-1).tolist()
    print([model.config.id2label[x] for x in preds])
