# -*- coding: utf-8 -*-
"""
使用SFTTrainer训练，应用chat template
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TokenizersBackend, EvalPrediction
from peft import get_peft_model, LoraConfig, PeftModel, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset


def build_prompt(shop_name: str, shop_brand: str, laiyuan1_id: str, brand_id: str, brand_name: str, _tokenizer_: TokenizersBackend):
    prompt_system = "假设你是一位品牌分类大师，请根据 shop_name, shop_brand, laiyuan1_id, brand_id，提取样本的品牌名称。请仅给出最终的品牌名称，而忽略其他无关的文字。"
    prompt_user = f"""
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
""".strip()
    prompt_assistant = brand_name
    return [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_user},
        {"role": "assistant", "content": prompt_assistant}
    ]


def _prepare_data_(df: DataFrame, _tokenizer_: TokenizersBackend):
    _result_ = df.fillna("")
    _result_["messages"] = _result_.apply(lambda ser: build_prompt(ser["shop_name"], ser["shop_brand"], ser["laiyuan1_id"], ser["brand_id"], ser["brand_name"], _tokenizer_), axis=1)
    return _result_


def compute_metrics(eval_pred: EvalPrediction):
    y_pred = eval_pred.predictions[:, :-1]
    y_true = eval_pred.label_ids[:, 1:]
    y_pred = np.where(y_true != -100, y_pred, -100)
    return {"accuracy": float(np.all(y_true == y_pred, axis=1).sum() / y_true.shape[0])}


if __name__ == '__main__':
    # 加载预训练模型
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    # 千问tokenizer的chat template不兼容trl的messages写法，需要替换
    tokenizer.chat_template = tokenizer.chat_template.replace(
        "{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}",
        "{{'<｜Assistant｜>'}}{% generation %}{{content + '<｜end▁of▁sentence｜>'}}{% endgeneration %}"
    )
    # 千问模型没有pad token，需手动设置
    tokenizer.add_special_tokens({"pad_token": "<｜Pad｜>"})
    model.resize_token_embeddings(len(tokenizer))
    model.pad_token = tokenizer.pad_token
    model.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

    # 加载训练数据集
    data = pd.read_excel("data/2.1train.xlsx", index_col="id", dtype=str)
    data = _prepare_data_(data, tokenizer)
    data_train = Dataset.from_pandas(data, preserve_index=False)
    train_val = data_train.train_test_split(test_size=0.2, shuffle=True, seed=42)
    print(train_val)

    # 加载训练超参数
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    train_args = SFTConfig(
        "checkpoints",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        num_train_epochs=10,
        learning_rate=5e-4,
        lr_scheduler_type="reduce_lr_on_plateau",
        optim="adamw_torch_fused",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="best",
        save_steps=25,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        report_to="tensorboard",
        seed=42,
        data_seed=42,
        assistant_only_loss=True,
        # packing=True
    )
    trainer = SFTTrainer(
        model,
        args=train_args,
        train_dataset=train_val["train"],
        eval_dataset=train_val["test"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=lambda _logits_, _labels_: _logits_.argmax(dim=-1),
        peft_config=lora_config
    )
    trainer.train()
    trainer.save_model("adapter")
