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


class Preprocessor():
    def __init__(self, clip_tokenizer, llm_tokenizer):
        self.clip_tokenizer = clip_tokenizer
        self.llm_tokenizer = llm_tokenizer

    @staticmethod
    def infer_seqlen(source_len: int, target_len: int, cutoff_len: int):
        if target_len * 2 < cutoff_len:  # truncate source
            max_target_len = cutoff_len
        elif source_len * 2 < cutoff_len:  # truncate target
            max_target_len = cutoff_len - source_len
        else:  # truncate both
            max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

        new_target_len = min(max_target_len, target_len)
        max_source_len = max(cutoff_len - new_target_len, 0)
        new_source_len = min(max_source_len, source_len)
        return new_source_len, new_target_len

    def __call__(self, examples):
        model_inputs = {"input_ids": list(), "labels": list()}
        for i in range(len(examples["input"])):
            query, answer = examples["input"][i], examples["label"][i]
            input_ids = self.clip_tokenizer.encode(text=query, add_special_tokens=False)
            label_ids = self.llm_tokenizer.encode(text=answer, add_special_tokens=False)
            source_len, target_len = self.infer_seqlen(len(input_ids), len(label_ids), 4096)
            input_ids, label_ids = input_ids[:source_len], label_ids[:target_len]
            input_ids = input_ids + [self.llm_tokenizer.bos_token_id] + label_ids + [self.llm_tokenizer.eos_token_id]
            context_length = input_ids.index(self.llm_tokenizer.bos_token_id)
            label_ids = [-100] * context_length + input_ids[context_length:]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(label_ids)
        
        return model_inputs


class NewLLaVATrainer(Trainer):
    def __init__(self, clip_tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.clip_tokenizer = clip_tokenizer

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.clip_model.save_pretrained(os.path.join(output_dir, "clip_model"), state_dict=self.model.clip_model.state_dict(), safe_serialization=False)
        self.model.language_model.save_pretrained(os.path.join(output_dir, "language_model"), state_dict=self.model.language_model.state_dict(), safe_serialization=False)

        torch.save(self.model.multi_model_projector.state_dict(), os.path.join(output_dir, "multi_model_projector.pth"))

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(os.path.join(output_dir, "language_model"))
        if self.clip_tokenizer is not None:
            self.clip_tokenizer.save_pretrained(os.path.join(output_dir, "clip_model"))

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


class NLPTrainer:

    def __init__(
            self,
            clip_model_name="/root/autodl-tmp/jina-clip-v1",
            llm_model_name="/root/autodl-tmp/Meta-Llama-3.1-8B",
            output_dir="/root/autodl-tmp/robot_script",
            cache_dir="/root/autodl-tmp/huggingface",
            pre_set="fp16",
            use_deepspeed=True,
    ):
        self.clip_model_name = clip_model_name
        self.llm_model_name = llm_model_name

        self.output_dir = output_dir
        self.cache_dir = cache_dir

        self.args_fix = dict(
            # evaluation_strategy="epoch",       # 每个epoch模型推理验证一次
            learning_rate=1e-5,                # 学习率
            weight_decay=0.1,                  # 参数正则化比例
            lr_scheduler_type="cosine",        # 学习率预热优化器选择
            warmup_ratio=0.1,                  # 在训练数据steps总数中, 使用优化器的比例
            save_strategy="epoch",             # 模型每个epoch保存一次checkpoint
            no_cuda=False,                     # 是否不使用cuda
            seed=100,                          # 随机种子设定
            dataloader_num_workers=10,         # 数据加载使用的进程数量
            past_index=-1,                     # 模型推理时使用过去状态长度
            remove_unused_columns=False,       # 是否删除无用数据列
            group_by_length=False,             # 是否将训练数据集中长度大致相同的样本分到同一组
            include_inputs_for_metrics=False,  # 是否将输入传递给Trainer的函数compute_metrics
            logging_steps=0.01,                # 模型训练时,每多少步打印一次日志
        )

        if pre_set == "fp16":
            self.mix_pre = dict(
                fp16=True,                     # 是否使用float16(混合)精度训练
                fp16_full_eval=True,           # 是否使用float16(混合)精度推理验证
            )
        elif pre_set == "bf16":
            self.mix_pre = dict(
                bf16=True,                     # 是否使用float16(混合)精度训练
                # bf16_full_eval=True,           # 是否使用float16(混合)精度推理验证
            )
        else:
            self.mix_pre = dict()

        if use_deepspeed:
            self.deepspeed_set = dict()
        else:
            self.deepspeed_set = dict(
                load_best_model_at_end=True,   # 训练结束时加载训练期间找到的最佳模型
                gradient_checkpointing=True,   # 是否使用梯度检查点节省显存
                auto_find_batch_size=True,     # 当大batch size显存超了时候,自动降低batch size
            )

        self.args_dynamic = dict(
            output_dir=output_dir,             # 输出文件夹
            per_device_train_batch_size=2,    # 训练时每个GPU的batch size
            # per_device_eval_batch_size=16,   # 验证时每个GPU的batch size
            gradient_accumulation_steps=1,     # 几个steps更新一次参数
            num_train_epochs=30,               # 训练epoch数量
            ignore_data_skip=False,            # 训练恢复时,是否跳过之前的训练步骤
            neftune_noise_alpha=None,          # 指令微调时,可以对Embedding加噪提升效果,默认设置为5
            save_only_model=True,              # 不存储中间变量,这其实也属于放弃训练恢复
            report_to="none",                  # 不将结果上传至wandb
        )

    def trainer(
            self,
            train_file_path="/root/autodl-tmp/dataset/alpaca-cleaned",
    ):
        train_arg = TrainingArguments(
            **self.args_fix, **self.args_dynamic, **self.mix_pre, **self.deepspeed_set
        )

        clip_tokenizer = AutoTokenizer.from_pretrained(self.clip_model_name, cache_dir=self.cache_dir, use_fast=False)
        llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name, cache_dir=self.cache_dir, use_fast=False)

        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token

        model = LlavaForConditionalGeneration(self.clip_model_name, self.llm_model_name, cache_dir=self.cache_dir, special_token_id=llm_tokenizer.bos_token_id, clip_pad_token_id=clip_tokenizer.pad_token_id, llm_pad_token_id=llm_tokenizer.pad_token_id)

        train_dataset = load_dataset(train_file_path, split="train")
        preprocessor = Preprocessor(clip_tokenizer, llm_tokenizer)
        train_dataset = train_dataset.map(lambda example: {"input": example["instruction"]+example["input"], "label": example["output"]}, remove_columns=["instruction", "input", "output"])
        train_dataset = train_dataset.map(preprocessor, batched=True, remove_columns=["input", "label"])
        data_collator = DataCollatorForSeq2Seq(llm_tokenizer, pad_to_multiple_of=8, padding=True, label_pad_token_id=-100)

        trainer = NewLLaVATrainer(
            model=model,                       # 训练验证模型
            args=train_arg,                    # 训练验证参数
            data_collator=data_collator,       # 设置dataloader的collate_fn参数
            train_dataset=train_dataset,       # 训练数据集
            tokenizer=llm_tokenizer,           # 训练验证token
            clip_tokenizer=clip_tokenizer,
        )
        trainer.train()


if __name__ == "__main__":
    nlp_trainer = NLPTrainer()
    nlp_trainer.trainer()
    pass
