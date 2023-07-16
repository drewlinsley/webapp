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
from train import ClipCaptionModel, ClipCaptionPrefix, ClipCocoDataset, MappingType
from typing import Tuple, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig  # , MappingType
import selfies as sf
from glob import glob
from accelerate import Accelerator
from accelerate.logging import get_logger
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)


PRETRAINED = "ncfrey/ChemGPT-1.2B"
PRETRAINED = "facebook/galactica-1.3b"
CACHE_DIR = "/media/data_cifs/projects/prj_video_imagenet/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def train(accelerator, logger, train_dataset: ClipCocoDataset, test_dataset: ClipCocoDataset, model: ClipCaptionModel, args,
          lr: float = 5e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):  # 2e-5 and 5000 steps is the normal

    # device = accelerator.device  # torch.device('cuda:0')
    batch_size = args.bs
    test_batch_size = args.test_bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # model = model.to(device)
    # model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.00001)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=24, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, drop_last=True, num_workers=24, pin_memory=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * (len(train_dataloader) // batch_size)
    )
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    # )
    if args.state is not None:
        accelerator.load_state(args.state)

    if "galactica" in PRETRAINED.lower():
        ignore_index = 1
    else:
        import pdb;pdb.set_trace()
        ignore_index = 0

    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, test_dataloader, scheduler)
    weights = torch.load(args.ckpt, map_location="cpu")
    weights = {k.split("_orig_mod.")[1]: v for k, v in weights.items()}
    keys = [k for k, v in model.named_parameters()]
    both = [k for k in weights.keys() if k in keys]
    leftover =  [k for k in weights.keys() if k not in keys]
    print("Leftover: {}".format(leftover))
    print("(The head weight is shared, and appears here)")
    model.load_state_dict(weights)
    device = accelerator.device
    model.to(device)
    # avg_loss = torch.tensor(0).float().to(device)
    # model, model.clip_project, model.gpt, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(model, model.clip_project, model.gpt, optimizer, train_dataloader, test_dataloader, scheduler)
    # accelerator.wait_for_everyone()
    accelerator.register_for_checkpointing(scheduler)
    best_test = 100
    step = 0
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix, disable=not accelerator.is_local_main_process)

        # Evaluate on test set
        model.eval()
        # model.clip_project.eval() 
        # model.gpt2.eval()
        with torch.no_grad():
            avg_loss = 0
            for idx, (tokens, mask, prefix) in enumerate(test_dataloader):
                # tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
                import pdb;pdb.set_trace()
                outputs = model(tokens, prefix, mask)
                logits = outputs.logits[:, test_dataset.prefix_length - 1: -1]  # -1]  # .float()
                loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=ignore_index)
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
    parser.add_argument('--bs', type=int, default=10)  # 12
    parser.add_argument('--test_bs', type=int, default=32)  # 12
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    parser.add_argument('--state', type=str, default=None, help='ckpt path')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--lr', dest='lr', default=1e-4)
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--max_grad_norm', type=float, default=2)
    parser.add_argument('--ckpt', type=str)
    args = parser.parse_args()
    prefix_length = args.prefix_length
    assert args.train_data is not None, "Pass a path for your preprocessed training data."
    assert args.eval_data is not None, "Pass a path for your preprocessed eval data."

    accelerator = Accelerator(log_with="wandb", gradient_accumulation_steps=11)
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
        del d.f
        d.close()

    prefix_dim = 1024
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    if args.only_prefix:
        model = ClipCaptionPrefix(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        logger.info("Train only prefix", main_process_only=True)
    else:
        model = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        logger.info("Train both prefix and GPT", main_process_only=True)
        # sys.stdout.flush()

    train(accelerator, logger, train_dataset, test_dataset, model, args, output_dir=args.out_dir, output_prefix=args.prefix, lr=args.lr)
    # REMINDER: Changed ReLUs to GeLUs


if __name__ == '__main__':
    main()
