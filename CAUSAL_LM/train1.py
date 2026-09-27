# -*- coding: utf-8 -*-
"""
自定义提示词训练
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, EvalPrediction, TokenizersBackend
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from trl import DataCollatorForCompletionOnlyLM


def build_prompt(shop_name: str, shop_brand: str, laiyuan1_id: str, brand_id: str, brand_name: str, _tokenizer_: TokenizersBackend) -> str:
    _result_ = f"""
{_tokenizer_.bos_token}假设你是一位品牌分类大师，请根据 shop_name, shop_brand, laiyuan1_id, brand_id，提取样本的品牌名称。请仅给出最终的品牌名称，而忽略其他无关的文字。
### 规则如下
“brand_name”可信度优先级（由高到低）：
“brand_name”来自“laiyuan1_id”—>brand_id为“9999*****”品牌—>brand_id为99998888****品牌
“laiyuan1_id”的品牌：具有最高优先级，合并品牌时优先合并为来源1的品牌
brand_id为“9999*****”品牌：具有次优先级，若品牌名称不是来源1的品牌，则合并品牌时优先考虑品牌ID为9999*****的品牌
brand_id为“99998888*****”品牌：具有最低优先级，若品牌名称不是来源1的品牌且品牌ID不是9999*****的品牌，则将品牌合并为品牌ID为99998888****的品牌。
### 样例
{{'shop_name': '1200bookshop(嘉禾云门店)', 'shop_brand': '1200bookshop', 'laiyuan1_id': '是', 'brand_id': '9999285772'}} 答案是 '1200bookshop'
{{'shop_name': '168台球俱乐部', 'shop_brand': '168台球俱乐部', 'laiyuan1_id': '', 'brand_id': '999976735'}} 答案是 '168台球俱乐部'
{{'shop_name': '168台球棋牌连锁(昆明路店)', 'shop_brand': '168台球棋牌连锁', 'laiyuan1_id': '', 'brand_id': '99998888500391'}} 答案是 '168台球俱乐部'
### 输入数据
{{'shop_name': '{shop_name}', 'shop_brand': '{shop_brand}', 'laiyuan1_id': '{laiyuan1_id}', 'brand_id': '{brand_id}'}}
### 答案
""".lstrip()
    if brand_name:
        return _result_ + brand_name + _tokenizer_.eos_token
    else:
        return _result_


def _prepare_data_(df: DataFrame, _tokenizer_: TokenizersBackend):
    _result_ = df.fillna("")
    if "brand_name" not in _result_.columns:
        _result_["brand_name"] = ""
    _result_["full"] = _result_.apply(lambda ser: build_prompt(ser["shop_name"], ser["shop_brand"], ser["laiyuan1_id"], ser["brand_id"], ser["brand_name"], tokenizer), axis=1)
    return _result_


def compute_metrics(eval_preds: EvalPrediction):
    y_pred = eval_preds.predictions[:, :-1]
    y_true = eval_preds.label_ids[:, 1:]
    y_pred = np.where(y_true != -100, y_pred, -100)
    return {"accuracy": float(np.all(y_pred == y_true, axis=1).sum() / y_true.shape[0])}


if __name__ == '__main__':
    # 加载预训练模型
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    # 千问系列模型没有pad_token，需要手动指定
    tokenizer.add_special_tokens({"pad_token": "<｜Pad｜>"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.pad_token = tokenizer.pad_token
    tokenizer.padding_side = "right"            # decoder模型在CAUSAL_LM时需要用 right pad

    # 加载训练数据集
    data_train = pd.read_excel("data/2.1train.xlsx", index_col="id", dtype=str)
    data_train = _prepare_data_(data_train, tokenizer)
    data_test = pd.read_excel(r"data/2.1test.xlsx", index_col="id", dtype=str)
    data_test = _prepare_data_(data_test, tokenizer)
    data: Dataset = DatasetDict({
        "train": Dataset.from_pandas(data_train, preserve_index=False),
        "test": Dataset.from_pandas(data_test, preserve_index=False)
    })["train"]
    data = data.map(lambda x: tokenizer(x["full"], padding=False, truncation=True), batched=True).select_columns(["input_ids", "attention_mask"])
    train_val = data.train_test_split(test_size=0.2, shuffle=True, seed=42)
    print(train_val)

    # 加载训练参数
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    training_args = TrainingArguments(
        "checkpoints",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        num_train_epochs=5,
        learning_rate=5e-4,
        lr_scheduler_type="reduce_lr_on_plateau",
        optim="adamw_torch_fused",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="best",
        save_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="tensorboard",
        seed=42,
        data_seed=42
    )
    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_val["train"],
        eval_dataset=train_val["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorForCompletionOnlyLM(response_template="### 答案\n", tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=lambda _logits_, _labels_: _logits_.argmax(dim=-1)
    )
    trainer.train()
    trainer.save_model("adapter")
