# -*- coding: utf-8 -*-
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, EvalPrediction
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, PeftModel, TaskType
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def compute_metrics(eval_pred: EvalPrediction):
    print(eval_pred.predictions.shape)
    y_pred = eval_pred.predictions.argmax(axis=-1)
    y_true = eval_pred.label_ids
    print(y_true.shape)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy_score(y_true, y_pred)
    }


if __name__ == '__main__':
    id2label = {
        0: "消极",
        1: "中性",
        2: "积极"
    }
    prompt = "请判断以下文字的感情色彩\n消极：0、中性：1、积极：2\n"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
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
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        modules_to_save=["score"]
    )
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    model = get_peft_model(model, lora_config)

    data = load_dataset("csv", data_files=r"data/train.csv", split="train")
    data = data.map(lambda x: tokenizer([prompt + xx for xx in x["text"]], padding=True, truncation=True), batched=True).rename_column("label", "labels").select_columns(["input_ids", "attention_mask", "labels"])
    train_val = data.train_test_split(test_size=0.2, shuffle=True, seed=42)
    data_train = train_val["train"]
    data_test = train_val["test"]
    train_args = TrainingArguments(
        output_dir="checkpoints",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        logging_steps=10,
        learning_rate=5e-4,
        lr_scheduler_type="reduce_lr_on_plateau",
        optim="adamw_torch_fused",
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="best",
        save_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="tensorboard",                        # tensorboard在中文路径下会有问题
        seed=42,
        data_seed=42
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=data_train,
        eval_dataset=data_test,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.model.config.pad_token_id = tokenizer.pad_token_id
    trainer.save_model("adapter")
