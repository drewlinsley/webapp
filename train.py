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
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_int8_training
# from openbabel import openbabel
from glob import glob
from accelerate import Accelerator
from accelerate.logging import get_logger
from functools import partial
from tqdm import tqdm as std_tqdm
import wandb


tqdm = partial(std_tqdm, dynamic_ncols=True)


TOKENS = {
    "galactica": {"bos_token_id": 0, "eos_token_id": 2, "pad_token_id": 1,},
}


# GALACTICA_START = "[START_I_SMILES]"
# GALACTICA_END = "[END_I_SMILES]"
GALACTICA_START = "[START_SMILES]"
GALACTICA_END = "[END_SMILES]"
PRETRAINED = "ncfrey/ChemGPT-1.2B"
# PRETRAINED = "ncfrey/ChemGPT-19M"
# PRETRAINED = "ncfrey/ChemGPT-4.7M"
# PRETRAINED = "facebook/galactica-120b"
PRETRAINED = "facebook/galactica-1.3b"
# PRETRAINED = "../smiles-gpt/checkpoints/benchmark-10m"
# PRETRAINED = "facebook/galactica-6.7b"
# PRETRAINED = "DeepChem/ChemBERTa-77M-MLM"
MODEL = "llm_weights/step_3000_eval_0.6366949081420898.pth"

myhost = os.uname()[1]
if "oscar" in myhost:
    CACHE_DIR = "/cifs/data/tserre_lrs/projects/prj_video_imagenet/hf_cache"
else:
    CACHE_DIR = "/media/data_cifs/projects/prj_video_imagenet/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        raise RuntimeError("Depreciated")
        tokens = self.captions_tokens[item]
        # if "galactica" in PRETRAINED:
        #     pad_tok = 1
        # else:
        #     pad_tok = -1
        pad_tok = 1
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
        return tokens, mask, prefix

    def __init__(self, d,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False, pca=False, fold="train", mn=None, mx=None):

        # self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        if "gpt" in PRETRAINED.lower():
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(PRETRAINED)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED, cache_dir=CACHE_DIR)  # , use_fast=False)  # JUST ADDED FAST=FALSE
        if self.tokenizer.pad_token_id is None:
            if "galactica" in PRETRAINED.lower():
                td = TOKENS["galactica"]
            else:
                raise NotImplementedError(PRETRAINED)
            self.tokenizer.pad_token_id = td["pad_token_id"]
            self.tokenizer.eos_token_id = td["eos_token_id"]
            self.tokenizer.bos_token_id = td["bos_token_id"]
        self.pad_token_id = self.tokenizer.pad_token_id
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
        morphology = (morphology - self.min) / (self.max - self.min)
        morphology = (morphology - 0.5) / 0.5

        morphology = morphology.astype(np.float32)

        # # L2 normalize each row
        # morphology = morphology / np.linalg.norm(morphology, axis=-1, keepdims=True)

        # Convert to a tensor
        morphology = torch.tensor(morphology)

        # Prepare for training
        self.prefixes = morphology.reshape(len(morphology), -1)
        self.captions = mols  # smiles
        captions_raw = np.copy(mols)  # smiles)

        self.max_seq_len = 80  # 96  # 128
        # if "smiles-gpt" in PRETRAINED:
        #     pad_tok = 0
        # else:
        #     pad_tok = self.tokenizer.pad_token_id  # 1

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

            tokens = self.tokenizer.encode(caption, return_tensors="pt")
            # tokens = [0] + tokens  # Add BOS token
            # tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
            # tokens = self.tokenizer.encode(caption)
            # tokens = torch.tensor(tokens, dtype=torch.int64)
            padding = self.max_seq_len - tokens.shape[0]
            if padding > 0:
                # tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
                tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) + self.pad_token_id))
            elif padding < 0:
                tokens = tokens[:self.max_seq_len]

            mask = tokens.gt(self.pad_token_id)  # mask is zero where we out of sequence
            # tokens[~mask] = 0
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


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.GELU):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
                layers.append(nn.LayerNorm(sizes[i + 1], elementwise_affine=True))
        self.model = nn.Sequential(*layers)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=lambda x: nnf.gelu(x, approximate="tanh"), dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.q_norm = nn.LayerNorm(dim_self, elementwise_affine=True)
        self.k_norm = nn.LayerNorm(dim_self * 2, elementwise_affine=True)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # # b n h dh
        # queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        queries = self.q_norm(self.to_queries(x)).reshape(b, n, self.num_heads, c // self.num_heads)
        # # b m 2 h dh
        # keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys_values = self.k_norm(self.to_keys_values(y)).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=lambda x: nnf.gelu(x, approximate="tanh"),
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=lambda x: nnf.gelu(x, approximate="tanh"), norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]

        # Concatenate start token
        # out = self.trans_norm(out)
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8, num_heads: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, num_heads, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding) * 1e-4, requires_grad=True)
        # self.trans_norm = nn.LayerNorm(dim_embedding) 


class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        if "galactica" in PRETRAINED.lower():
            if self.use_lora:
                embedding_text = self.gpt.model.model.decoder.embed_tokens(tokens)
            else:
                embedding_text = self.gpt.model.decoder.embed_tokens(tokens)
        elif "bert" in PRETRAINED.lower():
            embedding_text = self.gpt.roberta.embeddings.word_embeddings(tokens)
        elif "gpt" in PRETRAINED.lower():
            embedding_text = self.gpt.transformer.wte(tokens)
        else:
            embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        # embedding_cat = torch.cat((prefix_projections, start_token_id), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)

        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP, use_lora=True):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.use_lora = use_lora
        # self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        print("Loading LM")
        if "galactica" in PRETRAINED.lower():
            self.gpt = AutoModelForCausalLM.from_pretrained(PRETRAINED, cache_dir=CACHE_DIR)  # , torch_dtype=torch.float8)  # , torch_dtype=torch.float16)
            # self.gpt = AutoModelForCausalLM.from_pretrained(PRETRAINED, cache_dir=CACHE_DIR, load_in_8bit=True, device_map="sequential")
            # self.gpt = prepare_model_for_int8_training(self.gpt)

            if use_lora:

                target_modules = ["q_proj", "v_proj"]
                config = LoraConfig(
                    r=16, lora_alpha=32, target_modules=target_modules, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
                )
                self.gpt = get_peft_model(self.gpt, config)

            # self.gpt.gradient_checkpointing_enable()
            # self.START_SMILES = 21  # self.tokenizer.encode("[START_I_SMILES]")
            # self.CHECKED = False
        elif "gpt" in PRETRAINED.lower():
            self.gpt = GPT2LMHeadModel.from_pretrained(PRETRAINED, cache_dir=CACHE_DIR)
            # self.gpt = AutoModelForCausalLM.from_pretrained(PRETRAINED, cache_dir=CACHE_DIR)
        else:
            # config = AutoConfig.from_pretrained(PRETRAINED)
            # config.is_decoder = True
            # config.max_position_embeddings
            # config.max_position_embeddings = 2048
            self.gpt = AutoModelForCausalLM.from_pretrained(PRETRAINED, cache_dir=CACHE_DIR)  # , config=config)  # , torch_dtype=torch.float16)
        # self.gpt.to(dtype)
        if use_lora == False:
            for k, param in self.gpt.named_parameters():
                if 1:  # if "wte" not in k:  #  and "embed_tokens" not in k:  # DO NOT FINETUNE FALACTICA EMB
                # if "wte" not in k and "embed_tokens" not in k:  # DO NOT FINETUNE FALACTICA EMB
                    param.requires_grad = False
            self.gpt.eval()

        if "galactica" in PRETRAINED.lower():
            if use_lora:
                self.gpt_embedding_size = self.gpt.model.model.decoder.embed_tokens.weight.shape[1]
            else:
                self.gpt_embedding_size = self.gpt.model.decoder.embed_tokens.weight.shape[1]
            # self.START_SMILES = self.gpt.model.decoder.embed_tokens(torch.tensor(self.START_SMILES))[None, None]
        elif "gpt" in PRETRAINED.lower():
            self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
            # self.gpt = AutoModelForCausalLM.from_pretrained(PRETRAINED, cache_dir=CACHE_DIR)
        elif "bert" in PRETRAINED.lower():
            self.gpt_embedding_size = self.gpt.roberta.embeddings.word_embeddings.weight.shape[1]
        else:
            self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]

        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length)).half()
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                     clip_length, num_layers)  # .half()
        # self.clip_project.to(dtype)


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length)
    else:
        model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")
    return model, parser


def train(accelerator, logger, train_dataset: ClipCocoDataset, test_dataset: ClipCocoDataset, model: ClipCaptionModel, args, sampler, pad_token,
          lr: float = 5e-5, warmup_steps: int = 10, output_dir: str = ".", output_prefix: str = ""):  # 2e-5 and 5000 steps is the normal

    # device = accelerator.device  # torch.device('cuda:0')
    batch_size = args.bs
    test_batch_size = args.test_bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.00001)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # shuffle=True,
        drop_last=True,
        num_workers=24,
        pin_memory=True)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=24,
        pin_memory=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * (len(train_dataloader) // batch_size)
    )
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    # )
    if args.state is not None:
        accelerator.load_state(args.state)

    # if "galactica" in PRETRAINED.lower():
    #     ignore_index = 1
    # else:
    #     ignore_index = 0
    ignore_index = pad_token
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, test_dataloader, scheduler)
    device = accelerator.device
    # model.to(device)
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
                outputs = model(tokens, prefix, mask, labels=tokens)
                logits = outputs.logits[:, train_dataset.prefix_length - 1: -1]  # .float()
                # logits = outputs.logits[:, train_dataset.prefix_length: -1]  # .float()
                # FIND WHERE THE END_I_SMILES IS AND ONLY EVAL UNTIL THERE -- INDEX=22
                loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=ignore_index)
                # loss.backward()
                # avg_loss = accelerator.gather(loss.repeat(batch_size)).mean()
                # accelerator.backward(avg_loss)
                accelerator.backward(loss)
                #params = {k: v for k, v in model.named_parameters()}
                #print(params["clip_project.transformer.layers.1.mlp.fc2.weight"].grad)
                #os._exit(1)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # accelerator.unscale_gradients(optimizer=optimizer)
                optimizer.step()
                scheduler.step()
                # optimizer.zero_grad(set_to_none=True)
                # for param in model.parameters():
                #     param.grad = None
                avg_loss += loss
            step += 1

            # loss = loss.item()
            # avg_losses.append(loss)
            progress.set_postfix({"train_loss": loss})
            progress.update()
            # accelerator.log({"epoch": epoch, "training_loss": loss}, step=step)
        # avg_loss = np.mean([x.item() for x in avg_losses])
        avg_loss = avg_loss.item() / float(idx + 1)
        progress.set_postfix({"Average train loss": avg_loss})
        accelerator.log({"epoch": epoch, "training_loss": avg_loss})  # , step=step)

        # Evaluate on test set
        model.eval()
        # model.clip_project.eval() 
        # model.gpt2.eval()
        with torch.no_grad():
            avg_loss = 0
            for idx, (tokens, mask, prefix) in enumerate(test_dataloader):
                # tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
                if 1:  # with accelerator.accumulate(model):
                    outputs = model(tokens, prefix, mask, labels=tokens)
                    logits = outputs.logits[:, test_dataset.prefix_length - 1: -1]  # -1]  # .float()
                    # logits = outputs.logits[:, train_dataset.prefix_length: -1]  # .float()
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
    parser.add_argument('--prefix_length', type=int, default=30)
    parser.add_argument('--prefix_length_clip', type=int, default=30)
    parser.add_argument('--bs', type=int, default=16)  # 12
    parser.add_argument('--test_bs', type=int, default=32)  # 16
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default="transformer")  # 'mlp', help='mlp/transformer')
    parser.add_argument('--state', type=str, default=None, help='ckpt path')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--lr', dest='lr', default=5e-4)  # 5e-5)
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--max_grad_norm', type=float, default=10)
    args = parser.parse_args()
    prefix_length = args.prefix_length
    assert args.train_data is not None, "Pass a path for your preprocessed training data."
    assert args.eval_data is not None, "Pass a path for your preprocessed eval data."

    if args.log:
        accelerator = Accelerator(log_with="wandb")  # , gradient_accumulation_steps=8)
        accelerator.init_trackers("CLIP")
    else:
        accelerator = Accelerator()
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
        train_compounds = train_dataset.filtered_mols
        ucs, class_sample_count = np.unique(train_compounds, return_counts=True)
        weight = {u: 1. / c for u, c in zip(ucs, class_sample_count)}
        samples_weight = np.asarray([weight[t] for t in train_compounds])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
        del d.f
        d.close()

    prefix_dim = 1024  # 1280 * 5
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    if args.only_prefix:
        model = ClipCaptionPrefix(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        logger.info("Train only prefix", main_process_only=True)
    else:
        model = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)  # , start_token_id=start_token_id)
        logger.info("Train both prefix and GPT", main_process_only=True)
        # sys.stdout.flush()

    # model = torch.compile(model)
    train(accelerator, logger, train_dataset, test_dataset, model, args, sampler=sampler, output_dir=args.out_dir, output_prefix=args.prefix, lr=args.lr, pad_token=train_dataset.pad_token_id)
    # REMINDER: Changed ReLUs to GeLUs


if __name__ == '__main__':
    main()
