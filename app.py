import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
import skimage.io as io
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import PIL.Image
import selfies as sf
from enum import Enum
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import gradio as gr


# import torch
PRETRAINED = "ncfrey/ChemGPT-1.2B"

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

WEIGHTS_PATHS = {
    "coco": "coco_weights.pt",
    "conceptual-captions": "conceptual_weights.pt",
}

D = torch.device
CPU = torch.device("cpu")


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.silu, dropout=0.):
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
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
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

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.silu,
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
                 mlp_ratio: float = 2., act=nnf.silu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
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
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8, num_heads: int = 16):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, num_heads, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP, freeze=True):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        # self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        config = AutoConfig(PRETRAINED)
        self.gpt = AutoModelForCausalLM(config)  # .from_pretrained(PRETRAINED)
        if freeze:
            for param in self.gpt.parameters():
                param.requires_grad = False

        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                     clip_length, num_layers)


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def generate_beam(
    model,
    tokenizer,
    beam_size: int = 20,
    prompt=None,
    embed=None,
    entry_length=100,
    temperature=1.0,
    sample=False,
    stop_token: str = "[SEP]",
):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[-1]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1)
            if not sample and i == 0:
                logits = logits.log()
            elif sample and i > 0:
                logits = logits.log()
            else:
                pass
            if scores is None:
                if sample:
                    next_tokens = torch.multinomial(logits, num_samples=beam_size)
                    scores = torch.gather(logits, -1, next_tokens).log()
                else:
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
                if sample:
                    scores_sum_average = torch.exp(scores_sum_average)  # Convert back to probabilities
                    scores_sum_average = scores_sum_average.view(-1)
                    next_tokens = torch.multinomial(scores_sum_average, num_samples=beam_size)
                    scores_sum_average = torch.gather(scores_sum_average, -1, next_tokens).log()
                else:
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
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        sf.decoder(tokenizer.decode(output[: int(length)], skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(" ", ""))
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
    model,
    tokenizer,
    tokens=None,
    prompt=None,
    embed=None,
    entry_count=10,
    entry_length=120,  # maximum number of words
    top_p=0.6,
    temperature=1,
    sample=False,
    stop_token: str = "[SEP]",
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[-1]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
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
                if sample:
                    probs = nn.functional.softmax(logits, dim=-1).squeeze(0)
                    next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
                else:
                    next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = sf.decoder(tokenizer.decode(output_list, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(" ", ""))
            generated_list.append(output_text)

    return generated_list[0]


def main(
        target=["TARDBP"],  # TARDBP"],
        num_seqs=10,
        sample="Sample",
        beam_search="Greedy",

        weights_path="checkpoints/checkpoint.pt",
        prefix_length=10,
        prefix_dim=96,
        prefix_length_clip=10,
        num_layers=8,
        figsize = (4, 4),
        device=torch.device("cuda")):  # cuda")):

    sample = sample == "Sample"
    beam_search = beam_search == "Beam search"
    tokenizer = AutoTokenizer.from_pretrained("ncfrey/ChemGPT-1.2B")
    model = ClipCaptionModel(
        prefix_length,
        clip_length=prefix_length,
        prefix_size=prefix_dim,
        num_layers=num_layers,
        mapping_type=MappingType.Transformer)
    weights = torch.load(weights_path, map_location=CPU)
    keys = [k for k, v in model.named_parameters()]
    both = [k for k in weights.keys() if k in keys]
    leftover =  [k for k in weights.keys() if k not in keys]
    model.load_state_dict(weights)
    model = model.eval()
    model = model.to(device)

    preproc = np.load("preproc_for_test.npz")
    orf_morphology = preproc["target"]
    orfs = preproc["orfs"]

    prefixes = []
    for t in target:
        idx = np.where(orfs == t)[0][0]
        prefixes.append(torch.tensor(orf_morphology[[idx]]).to(device))
    prefixes = torch.concat(prefixes, 0).float()
    with torch.no_grad():
        prefix_embed = model.clip_project(prefixes).reshape(len(prefixes), prefix_length, -1)
    # https://discuss.huggingface.co/t/how-to-generate-a-sequence-using-inputs-embeds-instead-of-input-ids/4145/3
    # https://discuss.huggingface.co/t/how-to-generate-a-sequence-using-inputs-embeds-instead-of-input-ids/4145/3
    # https://huggingface.co/docs/transformers/internal/generation_utils
    # https://huggingface.co/docs/transformers/generation_strategies#contrastive-search

    # sample_a = model.gpt.generate(inputs_embeds=prefix_embed, num_beams=4, do_sample=True)
    # sample_b = model.gpt.generate(inputs_embeds=prefix_embed, penalty_alpha=0.6, top_k=4, max_new_tokens=100)
    preds, images = [], []
    for _ in tqdm(range(num_seqs), total=num_seqs, desc="Sequences"):
        if beam_search:
            pred = generate_beam(model, tokenizer, embed=prefix_embed, sample=sample)[0]
        else:
            pred = generate2(model, tokenizer, embed=prefix_embed, sample=sample)
        preds.append(pred)
        mol = Chem.MolFromSmiles(pred)

        # Generate a 2D depiction of the molecule using RDKit
        img = Draw.MolToImage(mol, size=figsize)

        # # Display the image using matplotlib
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()
        # plt.close("all")
        images.append(img)
    outputs = preds + images
    return outputs


title = "De novo SMols"
description = "Generate SMols from a phenotype"
article="In this webapp you can supply a target and generate small molecules that will have a desirable/compensatory response to a manipulation of that target."
options = ["TARDBP", "PKD2", "SNCA", "MAPT", "MECP2", "ERBB2"]
num_outputs = 5
gr.inputs.Dropdown(options)
gr.Radio(["Beam search", "Greedy"])
demo = gr.Interface(
    main,
    inputs=[
        gr.inputs.Dropdown(options),
        gr.Radio(["Beam search", "Greedy"]),
        gr.Radio(["Sample", "Deterministic"])
    ],
    outputs=["text"] + ["image"] * num_outputs,
    examples=[
        ["PKD2", "Greedy", "Sample"],
        ["TARDBP", "Greedy", "Sample"],
    ],
    title=title,
    description=description,
    article=article).launch(share=True)


if __name__ == "__main__":
    demo.launch()

