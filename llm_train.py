import torch
import numpy as np
from joblib import Parallel, delayed
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, get_scheduler, default_data_collator
# from openbabel import openbabel
from glob import glob
from accelerate import Accelerator
from accelerate.logging import get_logger
from functools import partial
from tqdm import tqdm as std_tqdm
from smilesgpt_tokenization import SMILESBPETokenizer
from transformers.testing_utils import CaptureLogger
import transformers
from itertools import chain
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy, DefaultDataCollator, DataCollatorForLanguageModeling
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_int8_training
from datasets import Dataset
import math
import evaluate
from smiles_gpt_tokenization import SMILESBPETokenizer
tqdm = partial(std_tqdm, dynamic_ncols=True)


# GALACTICA_START = "[START_I_SMILES]"
# GALACTICA_END = "[END_I_SMILES]"
GALACTICA_START = "[START_SMILES]"
GALACTICA_END = "[END_SMILES]"
PRETRAINED = "ncfrey/ChemGPT-1.2B"
PRETRAINED = "../smiles-gpt/checkpoints/benchmark-10m"
PRETRAINED = "facebook/galactica-1.3b"
PRETRAINED = "facebook/galactica-125m"
CACHE_DIR = "/media/data_cifs/projects/prj_video_imagenet/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class ClipCocoDataset(Dataset):
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "attn": self.attn_masks[idx]}

    def __init__(self, d,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False, pca=False, fold="train", mn=None, mx=None):

        # self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        if "gpt" in PRETRAINED:
            # self.tokenizer = PreTrainedTokenizerFast.from_pretrained(PRETRAINED)
            self.tokenizer = SMILESBPETokenizer.get_hf_tokenizer(os.path.join(PRETRAINED, "tokenizer.json"))
            # self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED, cache_dir=CACHE_DIR)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED, cache_dir=CACHE_DIR, bos_token='[CLS]', eos_token='[SEP]', pad_token='[PAD]')

        if fold == "train":
            mols = d["train_selfies"]
        else:
            mols = d["test_selfies"]

        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        max_length = max([len(x) for x in mols])
        max_length = 128

        self.input_ids = []
        self.attn_masks = []
        for caption in tqdm(mols, total=len(mols), desc="Processing {}".format(fold)):
            encodings_dict = self.tokenizer(caption, truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(encodings_dict['input_ids'])  # torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64))
            self.attn_masks.append(encodings_dict['attention_mask'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default=None)
    parser.add_argument('--eval_data', default=None)
    # parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--prefix', default='coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    parser.add_argument('--state', type=str, default=None, help='ckpt path')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--max_grad_norm', type=float, default=10)
    args = parser.parse_args()
    assert args.train_data is not None, "Pass a path for your preprocessed training data."
    assert args.eval_data is not None, "Pass a path for your preprocessed eval data."

    gradient_accumulation_steps = 1
    per_device_train_batch_size = 4  # 72
    per_device_eval_batch_size = 4  # 72
    num_warmup_steps = 100
    max_train_steps = 200000000
    checkpointing_steps = 500
    learning_rate = 1e-4
    use_lora = True

    ckpt_output_dir = "llm_weights_v2_{}".format(PRETRAINED.split("/")[1])
    if not os.path.exists(ckpt_output_dir):
        os.makedirs(ckpt_output_dir, exist_ok=True)
    # bos_token='[CLS]', eos_token='[SEP]', pad_token='[PAD]'

    if args.log:
        accelerator = Accelerator(log_with="wandb", gradient_accumulation_steps=gradient_accumulation_steps)  # , gradient_accumulation_steps=8)
        accelerator.init_trackers("CLIP")
    else:
        accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)

    logger = get_logger(__name__, log_level="DEBUG")  # INFO")

    d = np.load(args.train_data, allow_pickle=True)
    if "galactica" in PRETRAINED:
        data = d["train_mols"]
        umol = np.unique(data)
        print("Appending galactica tokens.")
        pdata = []
        for d in tqdm(umol, total=len(umol), desc="Adding [SMILES]"):
            pdata.append(GALACTICA_START + d + GALACTICA_END)
        umol = np.asarray(pdata)
        pad, bos, eos = "<pad>", "<s>", "</s>"
        text_column_name = "smiles"  # "selfies"
    elif "smiles-gpt" in PRETRAINED:
        pad, bos, eos = "<pad>", "<s>", "</s>"
        text_column_name = "smiles"  # "selfies"
        data = d["train_mols"]
        umol = np.unique(data)
    elif "ncfrey" in PRETRAINED:
        pad, bos, eos = "[PAD]", "[CLS]", "[SEP]"
        text_column_name = "selfies"  # "selfies"
        data = d["train_selfies"]
        umol = np.unique(data)
    else:
        raise NotImplementedError(PRETRAINED)

    # text_column_name = "selfies"  # "selfies"
    # data = d["train_selfies"]
    np.random.seed(42)
    test_ids = np.random.permutation(len(umol))[:1000]
    test_idx = np.in1d(np.arange(len(umol)), test_ids)
    test_set = umol[test_idx]
    train_set = umol[~test_idx]
    raw_train_dataset = Dataset.from_dict({text_column_name: train_set})
    raw_test_dataset = Dataset.from_dict({text_column_name: test_set})

    max_length = max([len(x) for x in data])
    if "smiles-gpt" in PRETRAINED:
        tokenizer = SMILESBPETokenizer.get_hf_tokenizer(os.path.join(PRETRAINED, "tokenizer.json"), max_length=512)
    else:
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED, cache_dir=CACHE_DIR, bos_token=bos, eos_token=eos, pad_token=pad)
    model = AutoModelForCausalLM.from_pretrained(PRETRAINED, cache_dir=CACHE_DIR)
    if use_lora:
        # target_modules = ["q_proj", "v_proj", "k_proj"]
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
        config = LoraConfig(
            r=16, lora_alpha=32, target_modules=target_modules, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)
    else:
        model.resize_token_embeddings(len(tokenizer))
    block_size = min(tokenizer.model_max_length, 1024)

    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples, max_length=max_length):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name], padding='max_length', truncation=True, max_length=max_length)
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output


    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()

    remove_column_name = [text_column_name]
    with accelerator.main_process_first():  # (desc="dataset map tokenization"):
        tokenized_train = raw_train_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=24,
            load_from_cache_file=False,
            remove_columns=remove_column_name,
            desc="Running tokenizer on dataset",
        )
        tokenized_test = raw_test_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=24,
            load_from_cache_file=False,
            remove_columns=remove_column_name,
            desc="Running tokenizer on dataset",
        )
        remove_column_name = ["token_type_ids"]
        train_dataset = tokenized_train.map(
            group_texts,
            batched=True,
            num_proc=24,
            load_from_cache_file=False,
            remove_columns=remove_column_name,
            desc=f"Grouping texts in chunks of {block_size}",
        )
        test_dataset = tokenized_test.map(
            group_texts,
            batched=True,
            num_proc=24,
            load_from_cache_file=False,
            remove_columns=remove_column_name,
            desc=f"Grouping texts in chunks of {block_size}",
        )

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    default_data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        test_dataset, collate_fn=default_data_collator, batch_size=per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 1e-6,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = 1000  # math.ceil(max_train_steps / num_update_steps_per_epoch)
    total_batch_size = per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    completed_steps = 0
    starting_epoch = 0
    best_loss = 100000
    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        total_loss = 0
        active_dataloader = train_dataloader
        pbar = tqdm(total=len(active_dataloader), disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(active_dataloader):
            completed_steps += 1
            with accelerator.accumulate(model):
                batch.pop("token_type_ids")
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            pbar.set_description("Loss: {}".format(loss))
            pbar.update(1)

            if completed_steps % checkpointing_steps == 0 and completed_steps > 0:
                model.eval()
                losses = []
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        batch.pop("token_type_ids")
                        outputs = model(**batch)

                    loss = outputs.loss
                    losses.append(accelerator.gather_for_metrics(loss.repeat(per_device_eval_batch_size)))

                losses = torch.cat(losses)
                try:
                    eval_loss = torch.mean(losses)
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    perplexity = float("inf")
                pbar.set_description("Eval loss: {}".format(eval_loss))
                print("Eval loss: {}".format(eval_loss))
                if eval_loss < best_loss:
                    output_dir = f"step_{completed_steps}_eval_{eval_loss}.pth"
                    output_dir = os.path.join(ckpt_output_dir, output_dir)
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    accelerator.save({"model": unwrapped_model.state_dict()}, output_dir)
                    best_loss = eval_loss

            #     if completed_steps % checkpointing_steps == 0:
            #         output_dir = f"step_{completed_steps }.pth"
            #         output_dir = os.path.join(ckpt_output_dir, output_dir)
            #         accelerator.wait_for_everyone()
            #         unwrapped_model = accelerator.unwrap_model(model)
            #         accelerator.save({"model": unwrapped_model.state_dict()}, output_dir)
            #         # accelerator.wait_for_everyone()
            # if completed_steps >= max_train_steps:
            #     break

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                batch.pop("token_type_ids")
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(per_device_eval_batch_size)))

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
        pbar.set_description("Eval loss: {}".format(eval_loss))

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")
        if eval_loss < best_loss:
            output_dir = f"step_{completed_steps}_eval_{eval_loss}.pth"
            output_dir = os.path.join(ckpt_output_dir, output_dir)
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save({"model": unwrapped_model.state_dict()}, output_dir)
            best_loss = eval_loss


if __name__ == '__main__':
    main()
