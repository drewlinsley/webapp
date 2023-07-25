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
# PRETRAINED = "facebook/galactica-1.3b"
PRETRAINED = "facebook/galactica-125m"
CACHE_DIR = "/media/data_cifs/projects/prj_video_imagenet/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


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

    def forward(self, input_ids: torch.Tensor, prefix: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, token_type_ids=None):
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
        # embedding_cat = torch.cat((prefix_projections, start_token_id), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(input_ids.shape[0], input_ids.device)
            labels = torch.cat((dummy_token, input_ids), dim=1)

        if len(attention_mask[0]) < len(labels[0]):
            diff = len(labels[0]) - len(attention_mask[0])
            dummy_token = torch.ones_like(attention_mask)[:, :diff]
            attention_mask = torch.cat((dummy_token, attention_mask), 1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=attention_mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP, use_lora=True, ckpt=None, tokenizer=None):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.use_lora = use_lora
        # self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        print("Loading LM")

        model = AutoModelForCausalLM.from_pretrained(PRETRAINED, cache_dir=CACHE_DIR)
        model.resize_token_embeddings(len(tokenizer))
        if use_lora:
            target_modules = ["q_proj", "v_proj", "k_proj"]
            # target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
            config = LoraConfig(
                r=16, lora_alpha=32, target_modules=target_modules, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, config)

        if ckpt is not None:
            kv = torch.load(ckpt, map_location=torch.device('cpu'))["model"]
            # kv = {k.replace("base_model.model.", ""): v for k, v in kv.items()}
            import pdb;pdb.set_trace()
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
    args = parser.parse_args()
    assert args.train_data is not None, "Pass a path for your preprocessed training data."
    assert args.eval_data is not None, "Pass a path for your preprocessed eval data."

    gradient_accumulation_steps = 1
    per_device_train_batch_size = 2  # 72
    per_device_eval_batch_size = 2  # 72
    num_warmup_steps = 100
    max_train_steps = 200000000
    checkpointing_steps = 10000
    learning_rate = 1e-5
    use_lora = True
    prefix_length = 30
    prefix_length_clip = 30
    prefix_dim = 1024  # 1280 * 5
    num_layers = 8
    mapping_type = "transformer"


    ckpt = "llm_weights/step_41500_eval_0.5636716485023499.pth"
    ckpt = "llm_weights_v2_galactica-125m/step_34500_eval_0.5711127519607544.pth"
    # ckpt_output_dir = "clipclap_llm_weights_v2"
    ckpt_output_dir = "clipclap_llm_weights_{}".format(PRETRAINED.split("/")[-1])
    # ckpt_output_dir = "clipclap_llm_weights_v2-125m"  # "clipclap_llm_weights_v2"  # llm_weights_v2_galactica-125m
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
            tedata.append(GALACTICA_START + d + GALACTICA_END)
        test_data = np.asarray(tedata)
        # umol = np.asarray(pdata)
        pad, bos, eos = "<pad>", "<s>", "</s>"
        text_column_name = "smiles"  # "selfies"
        morph_column_name = "prefix"  # morphology"
    else:
        raise NotImplementedError(PRETRAINED)

    # text_column_name = "selfies"  # "selfies"
    # data = d["train_selfies"]
    raw_train_dataset = Dataset.from_dict({text_column_name: train_data, morph_column_name: train_morph})
    raw_test_dataset = Dataset.from_dict({text_column_name: test_data, morph_column_name: test_morph})

    max_length = max([len(x) for x in train_data])
    if "smiles-gpt" in PRETRAINED:
        tokenizer = SMILESBPETokenizer.get_hf_tokenizer(os.path.join(PRETRAINED, "tokenizer.json"), max_length=512)
    else:
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED, cache_dir=CACHE_DIR, bos_token=bos, eos_token=eos, pad_token=pad)

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
        # train_dataset = tokenized_train.map(
        #     group_texts,
        #     batched=True,
        #     num_proc=24,  # 24,
        #     load_from_cache_file=False,
        #     remove_columns=remove_column_name,
        #     desc=f"Grouping texts in chunks of {block_size}",
        # )
        # test_dataset = tokenized_test.map(
        #     group_texts,
        #     batched=True,
        #     num_proc=24,
        #     load_from_cache_file=False,
        #     remove_columns=remove_column_name,
        #     desc=f"Grouping texts in chunks of {block_size}",
        # )
        train_dataset = tokenized_train
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
    loss_fct = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)

    # ckpt = "clipclap_llm_weights/step_120000_eval_0.3543463349342346.pth"
    # kv = torch.load(ckpt, map_location=torch.device('cpu'))["model"]
    # # kv = {k.replace("base_model.model.", ""): v for k, v in kv.items()}
    # import pdb;pdb.set_trace()
    # model.load_state_dict(kv)
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
                # loss = outputs.loss

                labels = batch["labels"]
                logits = outputs.logits[:, -labels.size(1) :, :]
                # Shift so that tokens < n predict n
                shift_logits = outputs.logits[..., prefix_length:-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)

                # Flatten the tokens
                # loss_fct = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, tokenizer.vocab_size), shift_labels.view(-1))

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
                        outputs = model(**batch)

                    # loss = outputs.loss
                    labels = batch["labels"]
                    logits = outputs.logits[:, -labels.size(1) :, :]
                    # Shift so that tokens < n predict n
                    shift_logits = outputs.logits[..., prefix_length:-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous().to(logits.device)

                    # Flatten the tokens
                    # loss_fct = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
                    loss = loss_fct(shift_logits.view(-1, tokenizer.vocab_size), shift_labels.view(-1))
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

            # loss = outputs.loss
            labels = batch["labels"]
            logits = outputs.logits[:, -labels.size(1) :, :]
            # Shift so that tokens < n predict n
            shift_logits = outputs.logits[..., prefix_length:-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device)

            # Flatten the tokens
            # loss_fct = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, tokenizer.vocab_size), shift_labels.view(-1))
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
