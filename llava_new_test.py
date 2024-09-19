import os
from datasets import load_dataset
from llava_arch_new import LlavaForConditionalGeneration
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModel, DataCollatorForSeq2Seq, PreTrainedModel
from typing import List, Optional
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
import torch
from PIL import Image
import requests
from transformers import AutoProcessor

def checkpoint_test(checkpoint_path=None):
    clip_model_path = os.path.join(checkpoint_path, "clip_model")
    language_model_path = os.path.join(checkpoint_path, "language_model")
    multi_model_projector_path = os.path.join(checkpoint_path, "multi_model_projector.pth")
    # clip_model_path = "/root/autodl-tmp/jina-clip-v1"
    # language_model_path = "/root/autodl-tmp/Meta-Llama-3.1-8B"

    cache_dir = "/root/autodl-tmp/huggingface"
    
    clip_tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/jina-clip-v1", cache_dir=cache_dir, use_fast=False)
    llm_tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/Meta-Llama-3.1-8B", cache_dir=cache_dir, use_fast=False)
    
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    processor = AutoProcessor.from_pretrained("/root/autodl-tmp/jina-clip-v1", cache_dir=cache_dir, trust_remote_code=True)
    # prompt = "Please tell me the content of the previous picture."
    prompt = "Give three tips for staying healthy."
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=image, text=prompt, return_tensors="pt", add_special_tokens=False)

    model = LlavaForConditionalGeneration(
        clip_model_path, language_model_path, multi_model_projector_path=multi_model_projector_path, 
        cache_dir=cache_dir, special_token_id=llm_tokenizer.bos_token_id, 
        clip_pad_token_id=clip_tokenizer.pad_token_id, llm_pad_token_id=llm_tokenizer.pad_token_id)
    
    inputs["pixel_values"] = None
    # Generate
    generate_ids = model.image_text_generation(**inputs, max_new_tokens=100)
    print(generate_ids)
    print(llm_tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0])

    pass


if __name__ == "__main__":
    # model = AutoModel.from_pretrained("/root/autodl-tmp/jina-clip-v1", cache_dir="/root/autodl-tmp/huggingface", trust_remote_code=True)
    # print(model)
    checkpoint_test("/root/autodl-tmp/robot_script/checkpoint-32350")
    pass
