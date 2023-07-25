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
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import DataStructs
from rdkit.Chem import AllChem
from smiles_gpt_tokenization import SMILESBPETokenizer
tqdm = partial(std_tqdm, dynamic_ncols=True)


# GALACTICA_START = "[START_I_SMILES]"
# GALACTICA_END = "[END_I_SMILES]"
GALACTICA_START = "[START_SMILES]"
GALACTICA_END = "[END_SMILES]"
PRETRAINED = "ncfrey/ChemGPT-1.2B"
PRETRAINED = "../smiles-gpt/checkpoints/benchmark-10m"
PRETRAINED = "facebook/galactica-1.3b"
CACHE_DIR = "/media/data_cifs/projects/prj_video_imagenet/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def compute_tanimoto_similarity(smiles1, smiles2):
    """
    Compute the Tanimoto similarity between two molecules.
    """
    m1 = Chem.MolFromSmiles(smiles1)
    m2 = Chem.MolFromSmiles(smiles2)
    fp1 = Chem.RDKFingerprint(m1)  # Chem.MolFromSmiles(smiles1))
    fp2 = Chem.RDKFingerprint(m2)  # Chem.MolFromSmiles(smiles2))
    sim = DataStructs.TanimotoSimilarity(fp1,fp2)
    # ffp1 = AllChem.GetMorganFingerprint(m1,2,useFeatures=True)
    # ffp2 = AllChem.GetMorganFingerprint(m2,2,useFeatures=True)
    # sim = DataStructs.DiceSimilarity(ffp1,ffp2)
    return sim


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


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
        # self.q_norm = nn.LayerNorm(dim_self, elementwise_affine=True)
        # self.k_norm = nn.LayerNorm(dim_self * 2, elementwise_affine=True)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # queries = self.q_norm(self.to_queries(x)).reshape(b, n, self.num_heads, c // self.num_heads)
        # # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        # keys_values = self.k_norm(self.to_keys_values(y)).reshape(b, m, 2, self.num_heads, c // self.num_heads)
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

    def forward(self, input_ids: torch.Tensor, prefix: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, embedding_cat: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, token_type_ids=None):
        if embedding_cat is None:
            if "galactica" in PRETRAINED.lower():
                if self.use_lora:
                    embedding_text = self.gpt.model.model.decoder.embed_tokens(input_ids)
                else:
                    embedding_text = self.gpt.model.decoder.embed_tokens(input_ids)
            elif "bert" in PRETRAINED.lower():
                embedding_text = self.gpt.roberta.embeddings.word_embeddings(input_ids)
            elif "gpt" in PRETRAINED.lower():
                embedding_text = self.gpt.transformer.wte(input_ids)
            else:
                embedding_text = self.gpt.transformer.wte(input_ids)

            prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
            embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)

        if labels is not None:
            dummy_token = self.get_dummy_token(input_ids.shape[0], input_ids.device)
            labels = torch.cat((dummy_token, input_ids), dim=1)

        # attention_mask = torch.ones_like(embedding_cat[..., 1])  # CHANGED THIS — WILL FUCK UP LOSS I THINK
        out = self.gpt(inputs_embeds=embedding_cat)  # , labels=labels, attention_mask=attention_mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP, use_lora=True, ckpt=None, tokenizer=None):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.use_lora = use_lora
        # self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        print("Loading LM")

        model = AutoModelForCausalLM.from_pretrained(PRETRAINED, cache_dir=CACHE_DIR)
        # model.resize_token_embeddings(len(tokenizer))
        if use_lora:
            target_modules = ["q_proj", "v_proj", "k_proj"]
            config = LoraConfig(
                r=16, lora_alpha=32, target_modules=target_modules, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, config)

        if ckpt is not None:
            kv = torch.load(ckpt, map_location=torch.device('cpu'))["model"]
            # kv = {k.replace("base_model.model.", ""): v for k, v in kv.items()}
            model.load_state_dict(kv)
        self.gpt = model

        if 1:  # use_lora == False:
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


def generate_beam(
    model,
    tokenizer,
    prefix,
    device,
    prefix_length,
    sample_tokens=True,
    lora_model=True,
    beam_size: int = 5,
    prompt="<s>" + GALACTICA_START,
    embed=None,
    entry_length=120,
    temperature=1.0,
    stop_token: str = GALACTICA_END,
):

    # model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    entry_count = len(prefix)
    sel_scores, sel_texts = [], []
    with torch.no_grad():
        prefix_projections = model.clip_project(prefix).view(len(prefix), prefix_length, -1)
        for entry_idx in tqdm(range(entry_count), total=entry_count, desc="Generating"):
            it_prefix = prefix_projections[entry_idx]
            seq_lengths = torch.ones(beam_size, device=device)
            is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
            scores = None
            tokens = None
            # if tokens is None:
            #     tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
            #     # tokens = tokens.unsqueeze(0).to(device)
            tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
            if lora_model:
                generated = model.gpt.model.model.decoder.embed_tokens(tokens)
            else:
                generated = model.gpt.model.decoder.embed_tokens(tokens)
            generated = torch.cat((it_prefix[None], generated), dim=1)

            for i in range(entry_length):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                logits = logits.softmax(-1).log()
                if scores is None:
                    scores, next_tokens = logits.topk(beam_size, -1)
                    generated = generated.expand(beam_size, *generated.shape[1:])
                    next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                    if tokens is None:
                        tokens = next_tokens
                    else:
                        tokens = tokens.expand(beam_size, *tokens.shape[1:])
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                else:
                    logits[is_stopped] = -float(np.inf)
                    logits[is_stopped, 0] = 0
                    scores_sum = scores[:, None] + logits
                    seq_lengths[~is_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                        beam_size, -1
                    )
                    next_tokens_source = next_tokens // scores_sum.shape[1]
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)
                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)
                    generated = generated[next_tokens_source]
                    scores = scores_sum_average * seq_lengths
                    is_stopped = is_stopped[next_tokens_source]
                if lora_model:
                    next_token_embed = model.gpt.model.model.decoder.embed_tokens(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
                else:
                    next_token_embed = model.gpt.model.decoder.embed_tokens(next_tokens.squeeze()).view(generated.shape[0], 1, -1)

                generated = torch.cat((generated, next_token_embed), dim=1)
                is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
                print(is_stopped)
                if is_stopped.all():
                    break
            scores = scores / seq_lengths
            output_list = tokens.cpu().numpy()
            output_texts = [
                tokenizer.decode(output[: int(length)], skip_special_tokens=True)
                for output, length in zip(output_list, seq_lengths)
            ]
            order = scores.argsort(descending=True)
            output_texts = [output_texts[i] for i in order]
            sel_texts.append(output_texts[0])
            sel_scores.append(scores.max())
    import pdb;pdb.set_trace()
    return sel_texts, sel_scores


def debug_generate2(
    model,
    tokenizer,
    prefix,
    device,
    dataloader,
    prefix_length,
    gt_tokens=None,
    prompt="<s>" + GALACTICA_START,
    # prompt=GALACTICA_START,
    embed=None,
    entry_count=1,
    entry_length=120,  # maximum number of words
    top_p=0.8,
    temperature=1.0,
    lora_model=True,
    sample_tokens=True,
    stop_token: str = GALACTICA_END,  # "</s>",
):
    # model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")

    entry_count = len(prefix)

    with torch.no_grad():
        # prefix_projections = model.clip_project(prefix).view(len(prefix), prefix_length, -1)
        for batch in tqdm(dataloader, total=entry_count, desc="Generating"):
            # it_prefix = prefix_projections[entry_idx]
            # if tokens is None:
            #     tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
            #     # tokens = tokens.unsqueeze(0).to(device)
            # tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
            # tokens = torch.concat((torch.zeros_like(gt_tokens[entry_idx])[0, [[0]]], gt_tokens[entry_idx]), 1)
            batch.pop("token_type_ids")
            outputs = model(**batch)
            # .gpt(inputs_embeds=generated, attention_mask=torch.ones_like(generated[[0], :, 0]))
            import pdb;pdb.set_trace()
            labels = batch["labels"]
            loss_fct = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
            logits = outputs.logits[:, -labels.size(1) :, :]
            # Shift so that tokens < n predict n
            shift_logits = outputs.logits[..., prefix_length:-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device)

            # Flatten the tokens
            loss = loss_fct(shift_logits.view(-1, tokenizer.vocab_size), shift_labels.view(-1))
 

def generate_beam_v2(
    model,
    tokenizer,
    embeds,
    device,
    lora_model,
    prefix_length,
    prompt="<s>" + GALACTICA_START,
    number_to_generate: int = 1,
    text_prefix_tokens: Optional[torch.Tensor] = None,
    beam_size: int = 20,
    entry_length: int = 80,
    temperature: float = 1.0,
    stop_token: str = GALACTICA_END
):

    stop_token = tokenizer.encode(stop_token, return_tensors="pt").to(device)[0]
    tokens = None
    scores = None

    seq_lengths = torch.ones(beam_size, device=embeds.device)
    has_stopped = torch.zeros(beam_size, dtype=torch.bool, device=embeds.device)
    generations, max_scores = [], []

    with torch.no_grad():
        # text_prefix_embed = model.language_model.get_embedding_text(text_prefix_tokens)
        # embeds = torch.cat((embeds, text_prefix_embed), dim=1)
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        if lora_model:
            generated = model.gpt.model.model.decoder.embed_tokens(tokens)
        else:
            generated = model.gpt.model.decoder.embed_tokens(tokens)
        prefix_projections = model.clip_project(embeds).view(len(embeds), prefix_length, -1)
        embeds = torch.cat((prefix_projections, generated), dim=1)

        for i in range(number_to_generate):
            for _ in range(entry_length):
                attention_mask = torch.ones_like(embeds[..., 0])  # torch.ones_like(embeds[0, :, [0]])
                outputs = model.gpt(inputs_embeds=embeds, attention_mask=attention_mask)

                # outputs = model.gpt(inputs_embeds=embeds)  # , attention_mask=attention_mask)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                logits = logits.softmax(-1).log()

                if scores is None:
                    scores, next_tokens = logits.topk(beam_size, -1)
                    embeds = embeds.expand(beam_size, *embeds.shape[1:])
                    next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                    if tokens is None:
                        tokens = next_tokens
                    else:
                        tokens = tokens.expand(beam_size, *tokens.shape[1:])
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                else:
                    logits[has_stopped] = -float(np.inf)
                    logits[has_stopped, 0] = 0  # stop_token[0]  # 0

                    scores_sum = scores[:, None] + logits
                    seq_lengths[~has_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)

                    next_tokens_source = torch.div(next_tokens, scores_sum.shape[1], rounding_mode='trunc')
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)

                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)

                    embeds = embeds[next_tokens_source]
                    scores = scores_sum_average * seq_lengths

                    import pdb;pdb.set_trace()
                    has_stopped = has_stopped[next_tokens_source]
                
                if lora_model:
                    next_token_embed = model.gpt.model.model.decoder.embed_tokens(next_tokens.squeeze())
                else:
                    next_token_embed = model.gpt.model.decoder.embed_tokens(next_tokens.squeeze())
                embeds = torch.cat((embeds, next_token_embed[:, None]), dim=1)
                # te = next_tokens.eq(stop_token).squeeze()
                # has_stopped[has_stopped != True] = te[has_stopped != True]
                has_stopped = torch.logical_or(has_stopped, next_tokens.eq(stop_token).squeeze())
                if has_stopped.all():
                    break

            scores = scores / seq_lengths
            output_list = tokens.cpu().numpy()
            output_texts = [tokenizer.decode(output[:int(length)], skip_special_tokens=True) for output, length in zip(output_list, seq_lengths)]

            order = scores.argsort(descending=True)
            output_texts = [output_texts[i] for i in order][0]
            max_scores.append(scores.max())
            generations.append(output_texts)
    return generations, max_scores


def generate2(
    model,
    tokenizer,
    prefix,
    device,
    prefix_length,
    tokens=None,
    prompt="<s>" + GALACTICA_START,
    # prompt=GALACTICA_START,
    embed=None,
    entry_count=1,
    entry_length=120,  # maximum number of words
    top_p=0.8,
    temperature=1.0,
    lora_model=True,
    sample_tokens=True,
    stop_token: str = GALACTICA_END,  # "</s>",
):
    # model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")

    entry_count = len(prefix)

    with torch.no_grad():
        prefix_projections = model.clip_project(prefix).view(len(prefix), prefix_length, -1)
        for entry_idx in tqdm(range(entry_count), total=entry_count, desc="Generating"):
            it_prefix = prefix_projections[entry_idx]
            # if tokens is None:
            #     tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
            #     # tokens = tokens.unsqueeze(0).to(device)
            tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
            if lora_model:
                generated = model.gpt.model.model.decoder.embed_tokens(tokens)
            else:
                generated = model.gpt.model.decoder.embed_tokens(tokens)
            generated = torch.cat((it_prefix[None], generated), dim=1)

            for i in range(entry_length):
                # model(tokens, prefix[[0]])
                attention_mask = torch.zeros_like(generated[..., 0])
                outputs = model.gpt(inputs_embeds=generated, attention_mask=attention_mask)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature)  #  if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                if sample_tokens:
                    next_token = torch.multinomial(nn.functional.softmax(logits, dim=-1).squeeze(0), num_samples=1).unsqueeze(0)
                else:
                    next_token = torch.argmax(logits, -1).unsqueeze(0)

                if lora_model:
                    next_token_embed = model.gpt.model.model.decoder.embed_tokens(next_token)
                else:
                    next_token_embed = model.gpt.model.decoder.embed_tokens(next_token)

                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list, skip_special_tokens=True)
            generated_list.append(output_text)
    return generated_list


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default=None)
    parser.add_argument('--eval_data', default=None)
    parser.add_argument('--out_dir', default='./checkpoints')
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
    parser.add_argument('--max_grad_norm', type=float, default=4)
    parser.add_argument('--target', type=str, default=None)

    args = parser.parse_args()
    assert args.train_data is not None, "Pass a path for your preprocessed training data."
    assert args.eval_data is not None, "Pass a path for your preprocessed eval data."
    inp_target = args.target

    gradient_accumulation_steps = 1
    per_device_eval_batch_size = 1
    use_lora = True
    prefix_length = 30
    prefix_length_clip = 30
    prefix_dim = 1024  # 1280 * 5
    num_layers = 8
    mapping_type = "transformer"
    ckpt_output_dir = "clipclap_llm_weights"
    # bos_token='[CLS]', eos_token='[SEP]', pad_token='[PAD]'

    if args.log:
        accelerator = Accelerator(log_with="wandb", gradient_accumulation_steps=gradient_accumulation_steps)  # , gradient_accumulation_steps=8)
        accelerator.init_trackers("CLIP")
    else:
        accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)

    logger = get_logger(__name__, log_level="DEBUG")  # INFO")

    d = np.load(args.train_data, allow_pickle=True)
    debug = False
    if "galactica" in PRETRAINED:
        train_data = d["train_mols"]
        test_data = d["test_mols"]
        train_morph = d["train_morphology"]
        test_morph = d["test_morphology"]

        if debug:
            train_data = train_data[:10000]
            train_morph = train_morph[:10000]
        print("Appending galactica tokens.")
        trdata = []
        for d in tqdm(train_data, total=len(test_data), desc="Adding [SMILES]"):
            trdata.append(GALACTICA_START + d + GALACTICA_END)
        train_data = np.asarray(trdata)

        tedata = []
        for d in tqdm(test_data, total=len(test_data), desc="Adding [SMILES]"):
            # tedata.append(GALACTICA_START + d + GALACTICA_END)
            tedata.append(GALACTICA_START + d + GALACTICA_END)
        test_data = np.asarray(tedata)
        # umol = np.asarray(pdata)
        pad, bos, eos = "<pad>", "<s>", "</s>"
        text_column_name = "smiles"  # "selfies"
        morph_column_name = "prefix"  # morphology"
    else:
        raise NotImplementedError(PRETRAINED)

    raw_test_dataset = Dataset.from_dict({text_column_name: test_data, morph_column_name: test_morph})
    max_length = max([len(x) for x in train_data])
    if "smiles-gpt" in PRETRAINED:
        tokenizer = SMILESBPETokenizer.get_hf_tokenizer(os.path.join(PRETRAINED, "tokenizer.json"), max_length=512)
    else:
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED, cache_dir=CACHE_DIR, bos_token=bos, eos_token=eos, pad_token=pad)

    ckpt = "llm_weights/step_41500_eval_0.5636716485023499.pth"
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    if args.only_prefix:
        model = ClipCaptionPrefix(prefix_length, clip_length=prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=num_layers, mapping_type=mapping_type, tokenizer=tokenizer, ckpt=ckpt)
        logger.info("Train only prefix", main_process_only=True)
    else:
        model = ClipCaptionModel(prefix_length, clip_length=prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=num_layers, mapping_type=mapping_type, tokenizer=tokenizer, ckpt=ckpt)  # , start_token_id=start_token_id)
        logger.info("Train both prefix and GPT", main_process_only=True)
        # sys.stdout.flush()

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
        result = examples
        result["labels"] = result["input_ids"].copy()
        attn = [[1] * prefix_length + x for x in result["attention_mask"]]
        result["attention_mask"] = attn


    remove_column_name = [text_column_name]
    with accelerator.main_process_first():  # (desc="dataset map tokenization"):
        tokenized_test = raw_test_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=24,
            load_from_cache_file=False,
            remove_columns=remove_column_name,
            desc="Running tokenizer on dataset",
        )
        remove_column_name = ["token_type_ids"]
        test_dataset = tokenized_test

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

    # Initialize our eval data as a control
    default_data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    eval_dataloader = DataLoader(
        test_dataset, collate_fn=default_data_collator, batch_size=per_device_eval_batch_size
    )

    targets = {
        "SOD1": "crispr_action",
        "PKD2": "ovr",
        "TARDBP": "ovr",
        "APP": "ovr",
        "MYC": "ovr",
        "ERBB2": "ovr",
        "CXCL1": "ovr",
        "SNCA": "ovr",
        "PSEN1": "ovr",
        "MAPT": "ovr",
        "rapamycin": "compounds",
        "DL-ALPHA-TOCOPHEROL": "compounds",
        "torin2": "compounds",
    }
    inchis = {
        "rapamycin": "InChI=1S/C56H87NO16/c1-33-17-13-12-14-18-34(2)45(68-9)29-41-22-20-39(7)56(67,73-41)51(63)52(64)57-24-16-15-19-42(57)53(65)71-46(30-43(60)35(3)26-38(6)49(62)50(70-11)48(61)37(5)25-33)36(4)27-40-21-23-44(47(28-40)69-10)72-54(66)55(8,31-58)32-59/h12-14,17-18,26,33,36-42,44-47,49-50,58-59,62,67H,15-16,19-25,27-32H2,1-11H3",
        "DL-ALPHA-TOCOPHEROL": "InChI=1S/C29H50O2/c1-20(2)12-9-13-21(3)14-10-15-22(4)16-11-18-29(8)19-17-26-25(7)27(30)23(5)24(6)28(26)31-29/h20-22,30H,9-19H2,1-8H3",
        "torin2": "InChI=1S/C24H15F3N4O/c25-24(26,27)17-2-1-3-18(11-17)31-22(32)9-6-16-13-29-20-7-4-14(10-19(20)23(16)31)15-5-8-21(28)30-12-15/h1-13H,(H2,28,30)"
    }

    version = 0.2
    # decode_type = "beam"
    # Load the genetic manipulation data
    data = np.load("smiles-gpt_controlled_preproc_for_test.npz")
    device = accelerator.device

    # Start with SOD1 to see if we generate antiox-like mols
    orfs, target_morph = data["orfs"], data["target"].astype(np.float32)
    crisprs, crispr_data = data["crisprs"], data["crispr_target"].astype(np.float32)
    limit = 10

    if inp_target is None:
        # Override with default
        sel_target = "torin2"  # "DL-ALPHA-TOCOPHEROL"  # "rapamycin"  # "PKD2"  # "PKD2"  # "SOD1"
        sel_target = "SOD1"
    else:
        sel_target = inp_target
    target_modality = targets[sel_target]

    if target_modality == "ovr":
        # target_idx = np.in1d(orfs, np.asarray(targets))
        target_idx = orfs == sel_target
        sel_morph = target_morph[target_idx]
        sel_orfs = orfs[target_idx]
        sel_morph = torch.from_numpy(sel_morph)
    elif target_modality == "crispr_action":
        # target_idx = np.in1d(orfs, np.asarray(targets))
        target_idx = crisprs == sel_target
        sel_morph = crispr_data[target_idx]
        sel_orfs = crisprs[target_idx]
        sel_morph = torch.from_numpy(sel_morph)
    elif target_modality == "crispr_action":
        # target_idx = np.in1d(orfs, np.asarray(targets))
        target_idx = crisprs == sel_target
        sel_morph = crispr_data[target_idx]
        sel_orfs = crisprs[target_idx]
        sel_morph = torch.from_numpy(sel_morph)
        wt_vector = np.median(crispr_data[crisprs == "non-targeting"], 0, keepdims=True)
        sel_morph = sel_morph - wt_vector  # Copy compound-action recipe. But apply to each sel_morph phenotype.
    elif target_modality == "compounds":
        data = np.load("smiles-gpt_controlled_preproc_for_train.npz", allow_pickle=True)
        if "antioxidants" in sel_target:
            pass
        else:
            # Load a compound and use its morphology
            comps = data["train_inchis"]
            train_morph = data["train_morphology"]
            target_idx = comps == inchis[sel_target]
            target_idx = np.where(target_idx)[0][:limit]
            sel_morph = train_morph[target_idx]
            sel_orfs = comps[target_idx]
            sel_morph = torch.from_numpy(sel_morph)
    else:
        raise NotImplementedError

    sel_morph = sel_morph.to(device)

    # Load new weights
    ckpt = "clipclap_llm_weights/step_120000_eval_0.3543463349342346.pth"
    kv = torch.load(ckpt, map_location=torch.device('cpu'))["model"]
    model.load_state_dict(kv)

    # Prepare everything with our `accelerator`.
    model, eval_dataloader, sel_morph = accelerator.prepare(model, eval_dataloader, sel_morph)

    loss_fct = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)

    # ckpt = "clipclap_llm_weights/step_120000_eval_0.3543463349342346.pth"
    # kv = torch.load(ckpt, map_location=torch.device('cpu'))["model"]
    # # kv = {k.replace("base_model.model.", ""): v for k, v in kv.items()}
    # import pdb;pdb.set_trace()
    # model.load_state_dict(kv)


    def get_dummy_token(batch_size: int, device: torch.device, prefix_length=30) -> torch.Tensor:
        return torch.zeros(batch_size, prefix_length, dtype=torch.int64, device=device)


    starting_epoch = 0
    max_tries = 10
    entry_length = 120
    temperature = 1.
    top_k = 0  # Strict threshold on top tokens
    top_p = 0.90  # Nucleus sampling
    sample_tokens = False  # True
    optimize_hits = False  # True  # True
    num_repeats = 20
    stop_token = GALACTICA_END
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    lora_model = True
    gpt_embedding_size = model.gpt.model.model.decoder.embed_tokens.weight.shape[1]
    max_val_count = 100
    beam_size = 10
    
    model.eval()

    gens, cumsum = [], []
    pre_embs, embs = [], []
    if not optimize_hits:
        start_token = tokenizer.encode(GALACTICA_START, return_tensors="pt").to(device)
        for morph, manip in tqdm(zip(sel_morph, sel_orfs), total=len(sel_morph), desc="Processing genetic manipulations"):
            with torch.no_grad():

                if lora_model:
                    generated = model.gpt.model.model.decoder.embed_tokens(start_token)  # [GALACTICA_START])
                else:
                    generated = model.gpt.model.decoder.embed_tokens(start_token)
                morph = morph[None]

                """
                seqs, scores = generate_beam(
                    model=model,
                    tokenizer=tokenizer,
                    prefix=morph,
                    device=device,
                    prefix_length=30,
                    beam_size=beam_size)
                import pdb;pdb.set_trace()
                gens.append(seqs)
                # gt_labels = gt_labels[gt_labels > 0]
                # gt_labels = tokenizer.decode(gt_labels, skip_special_tokens=True)
                # labs.append(gt_labels)
                cumsum.append(scores)

                """
                it_prefix = model.clip_project(morph).view(-1, 30, gpt_embedding_size)
                generated = torch.cat((it_prefix, generated), dim=1)

                probs = []  # Store the cumsum of logits = prod of probs
                tokens = tokenizer.encode(GALACTICA_START, return_tensors="pt").to(device)

                for i in range(entry_length):
                    # dummy_token = get_dummy_token(tokens.shape[0], device=accelerator.device)
                    # labels = torch.cat((dummy_token, tokens), dim=1)
                    # attention_mask = torch.ones_like(tokens)
                    # data = {"prefix": prefix, "input_ids": tokens, "labels": labels, "attention_mask": attention_mask}
                    data = {"embedding_cat": generated, "input_ids": tokens}  # , "attention_mask": attention_mask}
                    outputs = model(**data)  # .gpt(inputs_embeds=generated, attention_mask=attention_mask)
                    logits = outputs.logits
                    arg_token = nnf.softmax(logits[0, -1].view(-1), dim=0).max()
                    probs.append(arg_token.item())

                    logits = logits[:, -1, :] / (temperature)  #  if temperature > 0 else 1.0)
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        nnf.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = filter_value
                    if sample_tokens:
                        next_token = torch.multinomial(nn.functional.softmax(logits, dim=-1).squeeze(0), num_samples=1).unsqueeze(0)
                    else:
                        next_token = torch.argmax(logits, -1).unsqueeze(0)

                    if lora_model:
                        next_token_embed = model.gpt.model.model.decoder.embed_tokens(next_token)
                    else:
                        next_token_embed = model.gpt.model.decoder.embed_tokens(next_token)
                    generated = torch.cat((generated, next_token_embed), 1)

                    if tokens is None:
                        tokens = next_token
                    else:
                        tokens = torch.cat((tokens, next_token), dim=1)
                    if stop_token_index == next_token.item():
                        # final_emb = generated[0, prefix_length:].mean(0, keepdims=True).cpu().numpy()
                        final_emb = generated[0, prefix_length:].max(0)[0].detach().cpu().numpy()
                        break

                mu_prob = np.mean(probs)
                gens.append(tokenizer.decode(tokens.squeeze(), skip_special_tokens=True))
                cumsum.append(mu_prob)
                embs.append(final_emb)
                pre_embs.append(it_prefix.detach().cpu().numpy())
    else:

        start_token = tokenizer.encode(GALACTICA_START, return_tensors="pt").to(device)
        for morph, manip in tqdm(zip(sel_morph, sel_orfs), total=len(sel_morph), desc="Processing genetic manipulations"):

            with torch.no_grad():

                if lora_model:
                    original_gen = model.gpt.model.model.decoder.embed_tokens(start_token)  # [GALACTICA_START])
                else:
                    original_gen = model.gpt.model.decoder.embed_tokens(start_token)
                morph = morph[None]
                it_prefix = model.clip_project(morph).view(-1, 30, gpt_embedding_size)

                gt_tokens = tokenizer.encode(GALACTICA_START, return_tensors="pt").to(device)

            for _ in range(num_repeats):
                best_seq, best_prob, best_embs = [], [], []
                num_seqs = 0
                while True:
                    probs = []
                    generated = torch.cat((it_prefix, original_gen), dim=1)
                    tokens = gt_tokens.clone()

                    for i in range(entry_length):
                        # dummy_token = get_dummy_token(tokens.shape[0], device=accelerator.device)
                        # labels = torch.cat((dummy_token, tokens), dim=1)
                        # attention_mask = torch.ones_like(tokens)
                        # data = {"prefix": prefix, "input_ids": tokens, "labels": labels, "attention_mask": attention_mask}
                        data = {"embedding_cat": generated, "input_ids": tokens}  # , "attention_mask": attention_mask}
                        outputs = model(**data)  # .gpt(inputs_embeds=generated, attention_mask=attention_mask)
                        logits = outputs.logits
                        arg_token = nnf.softmax(logits[0, -1].view(-1), dim=0).max()
                        probs.append(arg_token.item())

                        logits = logits[:, -1, :] / (temperature)  #  if temperature > 0 else 1.0)

                        if top_k > 0:
                            if not isinstance(top_k, int) or top_k <= 0:
                                raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")
                            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
                            # Remove all tokens with a probability less than the last token of the top-k
                            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                            logits = logits.masked_fill(indices_to_remove, filter_value)

                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(
                            nnf.softmax(sorted_logits, dim=-1), dim=-1
                        )
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                            ..., :-1
                        ].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        logits[:, indices_to_remove] = filter_value
                        if sample_tokens:
                            next_token = torch.multinomial(nn.functional.softmax(logits, dim=-1).squeeze(0), num_samples=1).unsqueeze(0)
                        else:
                            next_token = torch.argmax(logits, -1).unsqueeze(0)

                        if lora_model:
                            next_token_embed = model.gpt.model.model.decoder.embed_tokens(next_token)
                        else:
                            next_token_embed = model.gpt.model.decoder.embed_tokens(next_token)
                        generated = torch.cat((generated, next_token_embed), 1)

                        if tokens is None:
                            tokens = next_token
                        else:
                            tokens = torch.cat((tokens, next_token), dim=1)
                        if stop_token_index == next_token.item():
                            break
                    # final_emb = generated[:, [-1]].detach().cpu().numpy()
                    # final_emb = generated[0, prefix_length + 1: -1].mean(0, keepdims=True).cpu().numpy()  # +1 and -1 on the prefix length to skip the SMILES-start/end tokens
                    final_emb = generated[0, prefix_length + 1: -1].max(0)[0].detach().cpu().numpy()  # +1 and -1 on the prefix length to skip the SMILES-start/end tokens
                    mu_prob = np.mean(probs)
                    print(mu_prob)
                    if mu_prob > 0.93:
                        break
                    best_seq.append(tokens)
                    best_prob.append(mu_prob)
                    best_embs.append(final_emb)
                    if num_seqs >= max_tries:
                        max_prob = np.argmax(best_prob)
                        tokens = best_seq[max_prob]
                        mu_prob = best_prob[max_prob]
                        emb = best_embs[max_prob]
                        break
                    else:
                        num_seqs += 1

                gens.append(tokenizer.decode(tokens.squeeze(), skip_special_tokens=True))
                cumsum.append(mu_prob)
                embs.append(emb)
                pre_embs.append(it_prefix.detach().cpu().numpy())

    df = pd.DataFrame(np.stack((gens, cumsum), 1), columns=["smiles", "confidence"])
    if "InChI" in manip:
        manip = sel_target
    df.to_csv("generated_{}_{}.csv".format(manip, target_modality))
    np.savez("generated_{}_{}.csv".format(manip, target_modality), smiles=gens, confidence=cumsum, embs=embs, pre_embs=pre_embs)

    # sort_idx = np.argsort(np.asarray([len(x) for x in gens]))
    sort_idx = np.argsort(cumsum)[::-1]
    sort_df = pd.DataFrame(np.stack((np.asarray(gens)[sort_idx], np.asarray(cumsum)[sort_idx]), 1), columns=["smiles", "confidence"])
    sort_df.to_csv("sorted_generated_{}_{}.csv".format(manip, target_modality))

    os._exit(1)  # Need to clean below this part
    import pdb;pdb.set_trace()
    compute_loss = True

    labs, gens = [], []
    cumsum, losses = [], []
    for step, batch in tqdm(enumerate(eval_dataloader), total=max_val_count, desc="Validating"):

        prefix = batch["prefix"]
        gt_labels = batch["labels"]
        tokens = batch["input_ids"]
        tokens = tokens[[0], [0]]
        tokens = tokens.reshape(1, -1)

        with torch.no_grad():
            if compute_loss:
                outputs = model(**batch)
                labels = batch["labels"]
                logits = outputs.logits[:, -labels.size(1) :, :]
                # Shift so that tokens < n predict n
                shift_logits = outputs.logits[..., prefix_length:-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)
                loss = loss_fct(shift_logits.view(-1, tokenizer.vocab_size), shift_labels.view(-1)).item()
                print("Loss = {}".format(loss))
                losses.append(loss)


        """
        prefix.requires_grad=True
        model.train()
        if lora_model:
            generated = model.gpt.model.model.decoder.embed_tokens(tokens)
        else:
            generated = model.gpt.model.decoder.embed_tokens(tokens)
        it_prefix = model.clip_project(prefix).view(-1, 30, gpt_embedding_size)
        generated = torch.cat((it_prefix, generated), dim=1)

        for i in range(entry_length):
            dummy_token = get_dummy_token(tokens.shape[0], device=accelerator.device)
            # labels = torch.cat((dummy_token, tokens), dim=1)
            attention_mask = torch.ones_like(tokens)
            data = {"prefix": prefix, "input_ids": tokens, "attention_mask": attention_mask}
            outputs = model(**data)  # .gpt(inputs_embeds=generated, attention_mask=attention_mask)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature)  #  if temperature > 0 else 1.0)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                nnf.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0
            logits = outputs.logits[:, -gt_labels.size(1) :, :]
            loss_1 = loss_fct(outputs.logits[0, -1].view(-1, tokenizer.vocab_size), gt_labels[0, i + 1][None])
            # accelerator.backward(loss_1)
            dc_da = torch.autograd.grad(loss_1, prefix, create_graph=True)
            loss_1.backward()
            loss_2 = ((prefix.grad - prefix) ** 2).mean()
            # accelerator.backward(loss_2)
            loss_2.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(loss_2)

        import pdb;pdb.set_trace()
        gg = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
        gt_labels = gt_labels[gt_labels > 0]
        gt_labels = tokenizer.decode(gt_labels, skip_special_tokens=True)

        os._exit()
        """


        with torch.no_grad():
            if compute_loss:
                outputs = model(**batch)

                labels = batch["labels"]
                logits = outputs.logits[:, -labels.size(1) :, :]
                # Shift so that tokens < n predict n
                shift_logits = outputs.logits[..., prefix_length:-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)
                loss = loss_fct(shift_logits.view(-1, tokenizer.vocab_size), shift_labels.view(-1))

            prefix = batch["prefix"]
            gt_labels = batch["labels"]
            tokens = batch["input_ids"]
            tokens = tokens[[0], [0]]
            tokens = tokens.reshape(1, -1)
            gt_tokens = tokens.clone()  # reshape(1, -1)

            if lora_model:
                generated = model.gpt.model.model.decoder.embed_tokens(tokens)
            else:
                generated = model.gpt.model.decoder.embed_tokens(tokens)
            it_prefix = model.clip_project(prefix).view(-1, 30, gpt_embedding_size)
            generated = torch.cat((it_prefix, generated), dim=1)

            probs = []  # Store the cumsum of logits = prod of probs
            for i in range(entry_length):
                # dummy_token = get_dummy_token(tokens.shape[0], device=accelerator.device)
                # labels = torch.cat((dummy_token, tokens), dim=1)
                # attention_mask = torch.ones_like(tokens)
                # data = {"prefix": prefix, "input_ids": tokens, "labels": labels, "attention_mask": attention_mask}
                data = {"embedding_cat": generated, "input_ids": tokens}  # , "attention_mask": attention_mask}
                outputs = model(**data)  # .gpt(inputs_embeds=generated, attention_mask=attention_mask)
                logits = outputs.logits
                arg_token = nnf.softmax(logits[0, -1].view(-1), dim=0).max()
                probs.append(arg_token.item())

                logits = logits[:, -1, :] / (temperature)  #  if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                if sample_tokens:
                    next_token = torch.multinomial(nn.functional.softmax(logits, dim=-1).squeeze(0), num_samples=1).unsqueeze(0)
                else:
                    next_token = torch.argmax(logits, -1).unsqueeze(0)

                if lora_model:
                    next_token_embed = model.gpt.model.model.decoder.embed_tokens(next_token)
                else:
                    next_token_embed = model.gpt.model.decoder.embed_tokens(next_token)
                generated = torch.cat((generated, next_token_embed), 1)

                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                if stop_token_index == next_token.item():
                    break

            mu_prob = np.mean(probs)
            gens.append(tokenizer.decode(tokens.squeeze(), skip_special_tokens=True))
            gt_labels = gt_labels[gt_labels > 0]
            gt_labels = tokenizer.decode(gt_labels, skip_special_tokens=True)
            labs.append(gt_labels)
            cumsum.append(mu_prob)
            if step >= max_val_count:
                break

            """
            best_seq, best_prob = [], []
            num_seqs = 0
            while True:
                probs = []
                generated = torch.cat((it_prefix, original_gen), dim=1)
                tokens = gt_tokens.clone()
                for i in range(entry_length):
                    dummy_token = get_dummy_token(tokens.shape[0], device=accelerator.device)
                    labels = torch.cat((dummy_token, tokens), dim=1)
                    attention_mask = torch.ones_like(tokens)
                    # data = {"prefix": prefix, "input_ids": tokens, "labels": labels, "attention_mask": attention_mask}
                    data = {"prefix": prefix, "input_ids": tokens}  # , "attention_mask": attention_mask}
                    outputs = model(**data)  # .gpt(inputs_embeds=generated, attention_mask=attention_mask)
                    logits = outputs.logits
                    arg_token = nnf.softmax(logits[0, -1].view(-1), dim=0).max()
                    probs.append(arg_token.item())

                    logits = logits[:, -1, :] / (temperature)  #  if temperature > 0 else 1.0)
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        nnf.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = filter_value
                    if sample_tokens:
                        next_token = torch.multinomial(nn.functional.softmax(logits, dim=-1).squeeze(0), num_samples=1).unsqueeze(0)
                    else:
                        next_token = torch.argmax(logits, -1).unsqueeze(0)

                    if lora_model:
                        next_token_embed = model.gpt.model.model.decoder.embed_tokens(next_token)
                    else:
                        next_token_embed = model.gpt.model.decoder.embed_tokens(next_token)

                    if tokens is None:
                        tokens = next_token
                    else:
                        tokens = torch.cat((tokens, next_token), dim=1)
                    if stop_token_index == next_token.item():
                        break
                mu_prob = np.mean(probs)
                print(mu_prob)
                if mu_prob > 0.9:
                    break
                best_seq.append(tokens)
                best_prob.append(mu_prob)
                if num_seqs >= max_tries:
                    max_prob = np.argmax(best_prob)
                    tokens = best_seq[max_prob]
                    mu_prob = best_prob[max_prob]
                    break
                else:
                    num_seqs += 1
            """
            # tokenizer.decode(tokens.squeeze(), skip_special_tokens=True);tokenizer.decode(labels[0][:25], skip_special_tokens=True)

        """
        seqs, scores = generate_beam_v2(
            model=model,
            tokenizer=tokenizer,
            embeds=prefix,
            device=device,
            lora_model=use_lora,
            prefix_length=30,
            beam_size=beam_size)
        gens.append(seqs)
        gt_labels = gt_labels[gt_labels > 0]
        gt_labels = tokenizer.decode(gt_labels, skip_special_tokens=True)
        labs.append(gt_labels)
        cumsum.append(scores)

        if step >= max_val_count:
            break
        """

    gens, labs, losses
    sims = []
    for l, g in tqdm(zip(labs, gens), total=len(labs), desc="Computing similarities"):
        try:
            sims.append(compute_tanimoto_similarity(g, l))
        except:
            print("Skipping {}".format(g))
            sims.append(np.nan)
    cumsum = np.asarray(cumsum)
    thresh = 0.9
    from matplotlib import pyplot as plt
    plt.plot(losses, label="losses")
    plt.plot(sims, label="similarties")
    plt.plot(cumsum, label="probs")
    # cumsum_y = cumsum[cumsum > thresh]
    # cumsum_x = np.arange(len(cumsum))[cumsum < thresh]
    # plt.stem(cumsum_x, cumsum_y, linefmt='grey', markerfmt='D')
    plt.title("Mean perf: {}, median perf: {}".format(np.nanmean(sims), np.nanmedian(sims)))
    plt.legend()
    plt.show()
    import pdb;pdb.set_trace()

    """
    for step, batch in tqdm(enumerate(eval_dataloader), total=max_val_count, desc="Validating"):
        # model.eval()
        stop_token_index = tokenizer.encode(stop_token)[0]
        entry_count = len(prefix)
        scores = None
        sel_texts = []
        with torch.no_grad():
            prefix = batch["prefix"]
            gt_labels = batch["labels"]
            tokens = batch["input_ids"]
            tokens = tokens[[0], [0]]
            tokens = tokens.reshape(1, -1)
            if lora_model:
                generated = model.gpt.model.model.decoder.embed_tokens(tokens)
            else:
                generated = model.gpt.model.decoder.embed_tokens(tokens)
            it_prefix = model.clip_project(prefix).view(-1, 30, gpt_embedding_size)
            generated = torch.cat((it_prefix, generated), dim=1)

            seq_lengths = torch.ones(beam_size, device=device)
            is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)

            for i in range(entry_length):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / temperature  # (temperature if temperature > 0 else 1.0)
                logits = logits.softmax(-1).log()
                if scores is None:
                    scores, next_tokens = logits.topk(beam_size, -1)
                    generated = generated.expand(beam_size, *generated.shape[1:])
                    next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                    if tokens is None:
                        tokens = next_tokens
                    else:
                        tokens = tokens.expand(beam_size, *tokens.shape[1:])
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                else:
                    logits[is_stopped] = -float(np.inf)
                    logits[is_stopped, 0] = 0
                    scores_sum = scores[:, None] + logits
                    seq_lengths[~is_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                        beam_size, -1
                    )
                    next_tokens_source = next_tokens // scores_sum.shape[1]
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)
                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)
                    generated = generated[next_tokens_source]
                    scores = scores_sum_average * seq_lengths
                    is_stopped = is_stopped[next_tokens_source]
                if lora_model:
                    next_token_embed = model.gpt.model.model.decoder.embed_tokens(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
                else:
                    next_token_embed = model.gpt.model.decoder.embed_tokens(next_tokens.squeeze()).view(generated.shape[0], 1, -1)

                generated = torch.cat((generated, next_token_embed), dim=1)
                is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
                if is_stopped.all():
                    break
            scores = scores / seq_lengths
            output_list = tokens.cpu().numpy()
            output_texts = [
                tokenizer.decode(output[: int(length)], skip_special_tokens=True)
                for output, length in zip(output_list, seq_lengths)
            ]
            order = scores.argsort(descending=True)
            output_texts = [output_texts[i] for i in order]
            sel_texts.append(output_texts[0])
            gt_labels = gt_labels[gt_labels > 0]
            gt_labels = tokenizer.decode(gt_labels, skip_special_tokens=True)

        if step >= max_val_count:
            break
    import pdb;pdb.set_trace()
    sims = []
    for l, g in tqdm(zip(labs, gens), total=len(labs), desc="Computing similarities"):
        try:
            sims.append(compute_tanimoto_similarity(g, l))
        except:
            print("Skipping {}".format(g))
    print("Mean Tanimoto similarity between generated and real molecules: {}".format(np.mean(sims)))
    print("Median Tanimoto similarity between generated and real molecules: {}".format(np.median(sims)))
    print("Max Tanimoto similarity between generated and real molecules: {}".format(np.max(sims)))
    """


if __name__ == '__main__':
    main()
