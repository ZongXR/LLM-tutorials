# -*- coding: utf-8 -*-
"""
使用chat template逐一样本推理
"""
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TokenizersBackend
from peft import get_peft_model, LoraConfig, PeftModel


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
    tokenizer = AutoTokenizer.from_pretrained("adapter")

    # 千问系列模型没有pad token，需手动补全
    tokenizer.add_special_tokens({"pad_token": "<｜Pad｜>"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token = tokenizer.pad_token
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"                             # decoder模型在推理时需使用 left pad

    # 加载测试数据
    data_test = pd.read_excel("data/2.1test.xlsx", index_col="id", dtype=str)
    # 加载adapter
    model = PeftModel.from_pretrained(model, "adapter")
    model.eval()
    with torch.no_grad():
        for idx, ser in data_test.iterrows():
            ser = ser.fillna("")
            ser["brand_name"] = ""
            full = build_prompt(**ser, _tokenizer_=tokenizer)
            inputs = tokenizer(full, padding=False, truncation=True, return_tensors="pt").to("cuda:0")
            generate_ids = model.generate(**inputs, do_sample=False)
            response = tokenizer.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(response)
