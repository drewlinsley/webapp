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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
# from openbabel import openbabel
from glob import glob
from accelerate import Accelerator
from accelerate.logging import get_logger
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)


# GALACTICA_START = "[START_I_SMILES]"
# GALACTICA_END = "[END_I_SMILES]"
GALACTICA_START = "[START_SMILES]"
GALACTICA_END = "[END_SMILES]"
PRETRAINED = "ncfrey/ChemGPT-1.2B"
# PRETRAINED = "ncfrey/ChemGPT-19M"
# PRETRAINED = "ncfrey/ChemGPT-4.7M"
# PRETRAINED = "facebook/galactica-120b"
# PRETRAINED = "facebook/galactica-1.3b"
# PRETRAINED = "facebook/galactica-125m"
# PRETRAINED = "../smiles-gpt/checkpoints/benchmark-10m"
# PRETRAINED = "facebook/galactica-6.7b"
# PRETRAINED = "DeepChem/ChemBERTa-77M-MLM"
CACHE_DIR = "/media/data_cifs/projects/prj_video_imagenet/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        # if "galactica" in PRETRAINED:
        #     pad_tok = 1
        # else:
        #     pad_tok = -1
        if "galactica" in PRETRAINED:
            pad_tok = 1
        elif "smiles-gpt" in PRETRAINED:
            pad_tok = 0
        else:
            raise NotImplementedError
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            # tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) + pad_tok))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()  # DEBUG
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        # if "galactica" in PRETRAINED:
        #     mask = torch.cat((mask[0][None], mask))
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        # tokens, mask = self.pad_tokens(item)
        tokens = self.captions_tokens[item]
        mask = self.masks[item]
        # prefix = self.prefixes[self.caption2embedding[item]]
        prefix = self.prefixes[item]  # .reshape(1, -1)
        prefix = prefix[None]
        return tokens, mask, prefix

    def __init__(self, d,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False, pca=False, fold="train", mn=None, mx=None):

        # self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        if "gpt" in PRETRAINED:
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(PRETRAINED)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED, cache_dir=CACHE_DIR)  # , use_fast=False)  # JUST ADDED FAST=FALSE
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix

        if fold == "train":
            morphology, mols = d["train_morphology"], d["train_mols"]
            self.min, self.max = morphology.min(0), morphology.max(0)
        else:
            morphology, mols = d["test_morphology"], d["test_mols"]
            # morphology = morphology[:6*8*10]
            # mols = mols[:6*8*10]
            self.min, self.max = mn, mx

        # Normalize to [-1, 1]
        # morphology = (morphology - self.min) / (self.max - self.min)
        # morphology = (morphology - 0.5) / 0.5

        morphology = morphology.astype(np.float32)

        # # L2 normalize each row
        # morphology = morphology / np.linalg.norm(morphology, axis=-1, keepdims=True)

        # Convert to a tensor
        morphology = torch.tensor(morphology)

        # Prepare for training
        self.prefixes = morphology.reshape(len(morphology), -1)
        self.captions = mols  # smiles
        captions_raw = np.copy(mols)  # smiles)

        self.max_seq_len = 128  # 128
        if "smiles-gpt" in PRETRAINED:
            pad_tok = 0
        else:
            pad_tok = 1

        self.captions_tokens = []
        self.caption2embedding = []
        self.masks = []
        # max_seq_len = 0
        assert len(captions_raw) == len(morphology), "Different number of compounds and morphologies."
        keep_morph = []
        for caption, emb in tqdm(zip(captions_raw, morphology), desc="Processing", total=len(morphology)):

            if len(caption) > self.max_seq_len:
                keep_morph.append(False)
                continue  # Remove these from training

            if "galactica" in PRETRAINED:
                caption = GALACTICA_START + caption + GALACTICA_END
            #     caption = caption + GALACTICA_END
            #     # caption = "[START_SMILES]" + caption + "[END_SMILES]"
            elif "smiles-gpt" in PRETRAINED:
                pass
                # caption = "<s>" + caption + "</s>"
            else:
                caption = "[CLS]" + caption

            tokens = self.tokenizer.encode(caption)
            # tokens = [0] + tokens  # Add BOS token
            tokens = torch.tensor(tokens, dtype=torch.int64)
            padding = self.max_seq_len - tokens.shape[0]
            if padding > 0:
                # tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
                tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) + pad_tok))
            elif padding < 0:
                tokens = tokens[:self.max_seq_len]

            mask = tokens.gt(pad_tok)  # mask is zero where we out of sequence
            tokens[~mask] = 0
            mask = mask.float()
            # mask = torch.cat((torch.ones(self.prefix_length + 1), mask), dim=0)  # adding prefix mask
            mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask

            self.captions_tokens.append(tokens)  # torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64))
            self.caption2embedding.append(emb)
            self.masks.append(mask)
            keep_morph.append(True)
        keep_morph = np.asarray(keep_morph)
        self.prefixes = self.prefixes[keep_morph]
        self.filtered_mols = captions_raw[keep_morph]


def train(accelerator, logger, train_dataset: ClipCocoDataset, test_dataset: ClipCocoDataset, model, args,
          lr, sampler, warmup_steps: int = 1000, output_dir: str = ".", output_prefix: str = "", nw=16, pm=True):  # 2e-5 and 5000 steps is the normal

    # device = accelerator.device  # torch.device('cuda:0')
    batch_size = args.bs
    test_batch_size = args.test_bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=nw, pin_memory=pm)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, drop_last=True, num_workers=nw, pin_memory=pm)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * (len(train_dataloader) // batch_size)
    # )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * (len(train_dataloader) // batch_size)
    )
    if args.state is not None:
        accelerator.load_state(args.state)

    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, test_dataloader, scheduler)
    device = accelerator.device
    model.to(device)

    # model.language_model.eval()
    # model.language_projection.train()
    # model.qformer.train()

    accelerator.register_for_checkpointing(scheduler)
    best_test = 100
    prev_loss = 0
    step = 0
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix, disable=not accelerator.is_local_main_process)
        # avg_losses = []
        model.train()
        # model.clip_project.train() 
        # model.gpt2.eval()
        # accelerator.wait_for_everyone()
        avg_loss = torch.tensor(0).float().to(device)  # avg_loss = 0.
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            # model.zero_grad()
            optimizer.zero_grad(set_to_none=True)
            # tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            mask = mask.to(prefix.device)
            with accelerator.accumulate(model):
                outputs = model(input_ids=tokens, image_embeds=prefix, labels=tokens)
                loss = outputs.loss
                # if loss - prev_loss > 0.5:
                #     print(idx, tokens)
                # prev_loss = loss
                accelerator.backward(loss)
                params = {k: v for k, v in model.named_parameters()}
                import pdb;pdb.set_trace()
                # print(params["clip_project.transformer.layers.1.mlp.fc2.weight"].grad)
                # os._exit(1)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                # optimizer.zero_grad(set_to_none=True)
                # for param in model.parameters():
                #     param.grad = None
                avg_loss += loss
            step += 1

            # loss = loss.item()
            # avg_losses.append(loss)
            progress.set_postfix({"train_loss": loss, "lr": scheduler.get_last_lr()[0]})
            progress.update()
            # accelerator.log({"epoch": epoch, "training_loss": loss}, step=step)
        # avg_loss = np.mean([x.item() for x in avg_losses])
        avg_loss = avg_loss.item() / float(idx + 1)
        progress.set_postfix({"Average train loss": avg_loss})
        accelerator.log({"epoch": epoch, "training_loss": avg_loss, "lr": scheduler.get_last_lr()[0]}, step=step)

        # Evaluate on test set
        # model.language_model.eval()
        # model.language_projection.eval()
        # model.qformer.eval()
        # model.clip_project.eval() 
        # model.gpt2.eval()
        model.eval()
        with torch.no_grad():
            avg_loss = 0
            for idx, (tokens, mask, prefix) in enumerate(test_dataloader):
                # tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
                outputs = model(input_ids=tokens, image_embeds=prefix, labels=tokens)
                loss = outputs.loss
                avg_loss += loss
                # avg_losses.append(loss)
                accelerator.log({"epoch": epoch, "eval_loss": loss}, step=step)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            avg_loss = avg_loss / float(idx + 1.)  # np.mean([x.item() for x in avg_losses])
            accelerator.log({"epoch": epoch, "average_eval_loss": avg_loss}, step=step)
            check_avg_loss = avg_loss.item()
            progress.set_postfix({"Average test loss": avg_loss})
            progress.close()
            if args.save_every > 0 and check_avg_loss < best_test:  # epoch % args.save_every == 0 or epoch == epochs - 1:
                best_test = check_avg_loss
                save_path = os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt")
                # accelerator.wait_for_everyone()
                # unwrapped_model = accelerator.unwrap_model(model)
                # accelerator.log("Saving {}".format(save_path), step=idx)
                # torch.save(unwrapped_model.state_dict(), save_path)
                # accelerator.save_state(output_dir=save_path)
                accelerator.save(accelerator.unwrap_model(model).state_dict(), save_path)
    accelerator.end_training()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default=None)
    parser.add_argument('--eval_data', default=None)
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--prefix', default='coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=20)
    parser.add_argument('--prefix_length_clip', type=int, default=20)
    parser.add_argument('--bs', type=int, default=64)  # 12
    parser.add_argument('--test_bs', type=int, default=2)  # 12
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    parser.add_argument('--state', type=str, default=None, help='ckpt path')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--lr', dest='lr', default=1e-4)
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--max_grad_norm', type=float, default=10)
    args = parser.parse_args()
    prefix_length = args.prefix_length
    assert args.train_data is not None, "Pass a path for your preprocessed training data."
    assert args.eval_data is not None, "Pass a path for your preprocessed eval data."


    accelerator = Accelerator(log_with="wandb", gradient_accumulation_steps=6)
    if args.log:
        accelerator.init_trackers("CLIP")
    logger = get_logger(__name__, log_level="DEBUG")  # INFO")
    with accelerator.main_process_first():
        d = np.load(args.train_data, allow_pickle=True)
        # logger.info("Building train data", main_process_only=True)
        print("Building train data")
        train_dataset = ClipCocoDataset(d, prefix_length, normalize_prefix=args.normalize_prefix, fold="train")
        mn, mx = train_dataset.min, train_dataset.max
        # logger.info("Building eval data", main_process_only=True)
        print("Building eval data")
        test_dataset = ClipCocoDataset(d, prefix_length, normalize_prefix=args.normalize_prefix, fold="eval", mn=mn, mx=mx)
        # train_compounds = d["train_mols"]
        train_compounds = train_dataset.filtered_mols
        ucs, class_sample_count = np.unique(train_compounds, return_counts=True)
        weight = {u: 1. / c for u, c in zip(ucs, class_sample_count)}
        samples_weight = np.asarray([weight[t] for t in train_compounds])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
        del d.f
        d.close()

    # start_token_id = train_dataset.tokenizer(GALACTICA_START, return_tensors="pt").input_ids  # .to(accelerator.device)
    # start_token_id = model.start_token

    # model = torch.compile(model)
    from modeling_blip_2 import Blip2ForConditionalGeneration
    from configuration_blip_2 import (
        Blip2Config,
    )
    # Initializing BLIP-2 vision, BLIP-2 Q-Former and language model configurations
    text_config = AutoConfig.from_pretrained(PRETRAINED)
    # config = Blip2Config.from_vision_qformer_text_configs(text_config=text_config)
    config = Blip2Config(text_config=text_config)
    model = Blip2ForConditionalGeneration(config)

    train(accelerator, logger, train_dataset, test_dataset, model, args, output_dir=args.out_dir, output_prefix=args.prefix, lr=args.lr, sampler=sampler)
    # REMINDER: Changed ReLUs to GeLUs


if __name__ == '__main__':
    main()
