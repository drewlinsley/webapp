import torch
import numpy as np
from joblib import Parallel, delayed
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
import selfies as sf
from openbabel import openbabel
from tqdm import tqdm
from glob import glob


PRETRAINED = "ncfrey/ChemGPT-1.2B"
# PRETRAINED = "facebook/galactica-120b"
# PRETRAINED = "facebook/galactica-6.7b"
CACHE_DIR = "/media/data_cifs/projects/prj_video_imagenet/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_smiles_from_inchi(inchi):
    """Convert smiles to inchi via openbabel.

    Thank you openai chatbot!"""
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("inchi", "smi")

    mol = openbabel.OBMol()
    obConversion.ReadString(mol, inchi)

    smiles = obConversion.WriteString(mol)
    return smiles.split("\t")[0]


def sf_decode(s):
    try:
        sm = sf.encoder(s)
    except:
        sm = False
    return sm


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        # prefix = self.prefixes[self.caption2embedding[item]]
        prefix = self.prefixes[item]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix

    def __init__(self, data_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        # self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED, cache_dir=CACHE_DIR)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix

        data = np.load("../linear_models/normalized_data-0.npz")
        #     np.savez("batch_normalized_prepped_data.npz".format(target), inchi=inchi, comp_data=comp_data, orf_data=orf_data, orfs=orfs)
        inchi = data["compounds"]
        morphology = data["comp_data"][:, None]
        # orfs = d["orfs"]
        # orf_data = d["orf_data"]

        # Convert to SMILES
        smiles = [get_smiles_from_inchi(x) for x in tqdm(inchi, total=len(inchi))]

        # Convert to selfies
        selfies, keeps = [], []
        for s in tqdm(smiles, total=len(smiles), desc="Converting smiles to selfies"):
            try:
                enc_sf = sf.encoder(s)
                selfies.append(enc_sf)
                keeps.append(True)
            except:
                keeps.append(False)
        selfies = np.asarray(selfies)
        keeps = np.asarray(keeps)
        morphology = morphology[keeps]

        # Truncate some of the crazier rows and normalize to [0, 1]
        thresh = 20
        mask = (np.abs(morphology.squeeze()) > thresh).sum(1) == 0
        print("Keeping {}/{} (removing {})".format(mask.sum(), len(mask), len(mask) - mask.sum()))
        selfies = selfies[mask]
        morphology = morphology[mask]
        # mm, mx = morphology.min(0), morphology.max(0)
        mm, mx = morphology.mean(0), morphology.std(0) + 1e-5  # -thresh, thresh
        # morphology = (morphology - mm) / (mx - mm)
        # morphology = (morphology - .5) / .5
        morphology = (morphology - mm) / mx

        target = data["orf_data"][:, None]
        orfs = data["orfs"]
        mask = (np.abs(target.squeeze()) > thresh).sum(1) == 0
        target = target[mask]
        target = (target - mm) / mx
        orfs = orfs[mask]
        # target = (target - mm) / (mx - mm)
        # target = (target - .5) / .5

        from sklearn.decomposition import PCA
        morphology = morphology.squeeze(1)
        target = target.squeeze(1)
        pca = PCA(n_components=96, whiten=True).fit(morphology)
        morphology = pca.transform(morphology)[:, None]
        target = pca.transform(target)[:, None]
        np.savez("preproc_for_test", mm=mm, mx=mx, thresh=thresh, target=target, orfs=orfs)

        #print([k for k in data.keys()])
        # m orphology = data["morphology"][:, None]
        # smiles = data["smiles"]
        # mask = (np.abs(morphology.squeeze()) > 5).sum(1) == 0  # Only take rows that are < 5SDs
        # print("Removing {} rows".format(len(smiles) - mask.sum()))
        # morphology = morphology[mask]
        # smiles = smiles[mask]
        # mx, mn = morphology.max(0), morphology.min(0)
        # morphology = (morphology - mn) / (mx - mn)
        # morphology = morphology.repeat(6, axis=2)
        morphology = torch.tensor(morphology)

        # # Convert smiles to selfies
        # selfies = Parallel(n_jobs=32)(delayed(sf_decode)(s) for s in tqdm(smiles, total=len(smiles), desc="Converting to selfies"))
        # mask = np.asarray([True if s else False for s in selfies])
        # selfies = np.asarray(selfies)[mask]
        # morphology = morphology[mask]
        self.prefixes = morphology
        # sf.decoder(self.tokenizer.decode(self.tokenizer.encode(sf.encoder(smiles[150])), skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(" ", ""))
        self.captions = selfies  # smiles
        captions_raw = np.copy(selfies)  # smiles)

        # with open(data_path, 'rb') as f:
        #     all_data = pickle.load(f)
        # print("Data size is %0d" % len(all_data["clip_embedding"]))
        # sys.stdout.flush()
        # self.prefixes = all_data["clip_embedding"]
        # captions_raw = all_data["captions"]
        # self.image_ids = [caption["image_id"] for caption in captions_raw]
        # self.captions = [caption['caption'] for caption in captions_raw]
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption, emb in tqdm(zip(captions_raw, morphology), desc="Processing", total=len(morphology)):
                # self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
                # self.caption2embedding.append(caption["clip_embedding"])
                if "galactica" in PRETRAINED:
                    caption = "[START_I_SMILES]" + caption + "[END_I_SMILES]"

                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64))
                self.caption2embedding.append(emb)
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            # self.max_seq_len = max_seq_len
            # print("Saving data")
            # with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
            #     pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))


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
        if "galactica" in PRETRAINED:
            embedding_text = self.gpt.model.decoder.embed_tokens(tokens)
        else:
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
        if "galactica" in PRETRAINED:
            self.gpt = AutoModelForCausalLM.from_pretrained(PRETRAINED, cache_dir=CACHE_DIR)  # , device_map="auto")  # , torch_dtype=torch.float16)
        else:
            self.gpt = AutoModelForCausalLM.from_pretrained(PRETRAINED, cache_dir=CACHE_DIR)
        if freeze:
            for k, param in self.gpt.named_parameters():
                if "wte" not in k:
                    param.requires_grad = False

        if "galactica" in PRETRAINED:
            self.gpt_embedding_size = self.gpt.model.decoder.embed_tokens.weight.shape[1]
        else:
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


def train(dataset: ClipCocoDataset, model: ClipCaptionModel, args,
          lr: float = 1e-4, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):  # 2e-5 is the normal

    from accelerate import Accelerator
    accelerator = Accelerator()

    device = accelerator.device  # torch.device('cuda:0')
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)  # , weight_decay=0.0001)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # datasets = load_data()
    # train_dataloader = DataLoader(train_dataloader["train"], batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)
    accelerator.wait_for_everyone()
    # accelerator.register_for_checkpointing(scheduler)

    # unwrapped_model = accelerator.unwrap_model(model)
    # torch.save(unwrapped_model.state_dict(), os.path.join(output_dir, f"{output_prefix}-pre.pt"))

    # accelerator.save_state(os.path.join(output_dir, f"{output_prefix}-pre.pt"))
    recent_saves = [os.path.join(output_dir, f"{output_prefix}-pre.pt")]
    recent_saves = []

    # save_config(args)
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            with accelerator.accumulate(model):
                outputs = model(tokens, prefix, mask)
                logits = outputs.logits[:, dataset.prefix_length - 1: -1]
                loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
                # loss.backward()
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            if 0:  # (idx + 1) % 10000 == 0:
                # accelerator.wait_for_everyone()
                # unwrapped_model = accelerator.unwrap_model(model)
                # torch.save(
                #     unwrapped_model.state_dict(),
                #     os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                # )
                save_path = os.path.join(output_dir, f"{output_prefix}_latest")
                accelerator.wait_for_everyone()
                # accelerator.save_state(save_path)

        progress.close()
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            # accelerator.wait_for_everyone()
            # unwrapped_model = accelerator.unwrap_model(model)
            # torch.save(
            #     unwrapped_model.state_dict(),
            #     os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            # )
            save_path = os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt")
            accelerator.wait_for_everyone()
            # accelerator.save_state(save_path)
            unwrapped_model = accelerator.unwrap_model(model)
            print("Saving {}".format(save_path))
            torch.save(unwrapped_model.state_dict(), save_path)
            recent_saves.append(save_path)
            if 0:  # len(recent_saves) > 2:
                print("Deleting {}".format(recent_saves[0]))
                files = sorted(glob(os.path.join(output_dir, "*")), key=os.path.getmtime)
                os.remove(files[0])
                recent_saves.pop(0)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/coco/oscar_split_train.pkl')
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--prefix', default='coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=6)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    parser.add_argument('--checkpoint', type=str, default=None, help='ckpt path')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    args = parser.parse_args()
    prefix_length = args.prefix_length
    dataset = ClipCocoDataset(args.data, prefix_length, normalize_prefix=args.normalize_prefix)
    prefix_dim = 96  # 640 if args.is_rn else 512
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    if args.only_prefix:
        model = ClipCaptionPrefix(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train only prefix")
    else:
        model = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train both prefix and GPT")
        sys.stdout.flush()

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
    train(dataset, model, args, output_dir=args.out_dir, output_prefix=args.prefix)
    # REMINDER: Changed ReLUs to GeLUs


if __name__ == '__main__':
    main()
