import os
import sys
import math
import torch
import wandb
import logging
import datasets
import argparse
import evaluate
import transformers

from typing import Optional
from itertools import chain
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from transformers.trainer_utils import get_last_checkpoint

# WandB 초기화
wandb.init(project='Hanghae99')
wandb.run.name = 'gpt-finetuning'

@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(default=None)  # HuggingFace hub에서 pre-trained 모델의 이름
    torch_dtype: Optional[str] = field(default=None, metadata={'choices': ['auto', 'bfloat16', 'float16', 'float32']})  # 모델의 precision
    
    dataset_name: Optional[str] = field(default=None)  # Fine-tuning에 사용할 dataset 이름
    dataset_config_name: Optional[str] = field(default=None)  # dataset configuration
    block_size: int = field(default=1024)  # input text의 길이
    num_workers: Optional[int] = field(default=None)  # 데이터 전처리에 사용할 worker 수
    validation_split_percentage: Optional[int] = field(default=10, metadata={"help": "Train/Validation split percentage"})  # Validation 데이터 비율

parser = HfArgumentParser((Arguments, TrainingArguments))
args, training_args = parser.parse_args_into_dataclasses()

logger = logging.getLogger()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if training_args.should_log:
    transformers.utils.logging.set_verbosity_info()  # log level을 INFO로 변경 

log_level = training_args.get_process_log_level()

# logger 설정
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

# HuggingFace logger 옵션 설정
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

logger.info(f"Training/evaluation parameters {training_args}")

# 데이터셋 로드
raw_datasets = load_dataset(
    args.dataset_name,
    args.dataset_config_name
)

# Train/Validation 분할
if "validation" not in raw_datasets.keys():
    raw_datasets = raw_datasets["train"].train_test_split(
        test_size=args.validation_split_percentage / 100,
        seed=42
    )
    raw_datasets["validation"] = raw_datasets.pop("test")

# 모델, 토크나이저 로드
config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    torch_dtype=args.torch_dtype
)

# 패딩 토큰 설정
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"

# 토크나이저 길이에 맞게 임베딩 크기 조정
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

# 데이터셋 컬럼 이름 설정
column_names = list(raw_datasets["train"].features)
text_column_name = "text" if "text" in column_names else column_names[0]

def tokenize_function(examples):
    output = tokenizer(examples[text_column_name])
    return output

with training_args.main_process_first(desc="dataset map tokenization"):
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=column_names
    )

max_pos_embeddings = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 1024
block_size = args.block_size if tokenizer.model_max_length is None else min(args.block_size, tokenizer.model_max_length)

def group_texts(examples):
    # 모든 텍스트를 하나로 합칩니다.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    
    # 전체 길이를 측정
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    
    # block_size로 텍스트 분할
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    
    # 라벨은 자기 자신
    result["labels"] = result["input_ids"].copy()
    return result

with training_args.main_process_first(desc="grouping texts together"):
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.num_workers
    )

# 학습 및 검증 데이터셋 설정
train_dataset = lm_datasets["train"]
eval_dataset = lm_datasets["validation"]

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Validation 데이터 추가
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

checkpoint = None
last_checkpoint = get_last_checkpoint(training_args.output_dir)
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
else:
    checkpoint = last_checkpoint

train_result = trainer.train(resume_from_checkpoint=checkpoint)

trainer.save_model()

metrics = train_result.metrics

# Validation metrics 계산
eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
metrics.update(eval_metrics)

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
