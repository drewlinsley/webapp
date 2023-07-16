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
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import PIL.Image
import selfies as sf
from enum import Enum
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import DataStructs
from rdkit.Chem import AllChem
import Levenshtein

import matplotlib.pyplot as plt
from train import GALACTICA_START, GALACTICA_END
from train import ClipCaptionModel, MappingType, ClipCaptionPrefix


# import torch
GALACTICA_START = "[START_I_SMILES]"
GALACTICA_END = "[END_I_SMILES]"
GALACTICA_START = "[START_SMILES]"
GALACTICA_END = "[END_SMILES]"

PRETRAINED = "ncfrey/ChemGPT-1.2B"
PRETRAINED = "facebook/galactica-1.3b"
CACHE_DIR = "/media/data_cifs/projects/prj_video_imagenet/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
CPU = torch.device("cpu")


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


def generate_beam(
    model,
    tokenizer,
    labels,
    beam_size: int = 20,
    prompt=None,
    embed=None,
    entry_length=150,
    temperature=1.0,
    sample=False,
):

    model.eval()
    if "galactica" in PRETRAINED.lower():
        stop_token = "[END_I_SMILES]"
    else:
        stop_token = "[SEP]"
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
        if "galactica" in PRETRAINED.lower():
            generated = torch.concat((generated, model.gpt.model.decoder.embed_tokens(torch.tensor(tokenizer.encode("[START_I_SMILES]")))[:, None]), 1)
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
                # next_tokens_source = torch.div(next_tokens, scores_sum.shape[1])
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]

            if "galactica" in PRETRAINED.lower():
                next_token_embed = model.gpt.model.decoder.embed_tokens(next_tokens.squeeze()).view(
                    generated.shape[0], 1, -1
                )
            elif "bert" in PRETRAINED.lower():
                next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                    generated.shape[0], 1, -1
                )
            else:
                import pdb;pdb.set_trace()
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    # test = sf.decoder(tokenizer.decode(output_list[0], seq_lengths[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(" ", ""))
    output_texts = []
    import pdb;pdb.set_trace()
    for output, length in zip(output_list, seq_lengths):
        if "galactica" in PRETRAINED.lower():
            end_pos = np.where((output == stop_token_index))[0][0]
            outp = tokenizer.decode(output[1:end_pos])
            output_texts.append(outp)
        else:
            output_texts.append(sf.decoder(tokenizer.decode(output[: int(length)], skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(" ", "")))
    order = scores.argsort(descending=True)
    import pdb;pdb.set_trace()
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
    model,
    tokenizer,
    labels,
    tokens=None,
    prompt=None,
    embed=None,
    entry_count=10,
    entry_length=150,  # maximum number of words
    top_p=0.8,
    temperature=1.,
    sample=False,
):

    model.eval()
    if "galactica" in PRETRAINED.lower():
        stop_token = GALACTICA_END
    else:
        stop_token = "[SEP]"
    stop_token_index = tokenizer.encode(stop_token)[-1]
    generated_list = []
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            it_sample = np.copy(sample)
            generated = embed.clone()
            tokens = None

            for i in range(entry_length):

                import pdb;pdb.set_trace()
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
                if it_sample:
                    probs = nn.functional.softmax(logits, dim=-1).squeeze(0)
                    next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
                else:
                    next_token = torch.argmax(logits, -1).unsqueeze(0)
                if stop_token_index == next_token.item():
                    it_sample = False  # Stop sampling after SMILES_END token is found

                if "galactica" in PRETRAINED.lower():
                    next_token_embed = model.gpt.model.decoder.embed_tokens(next_token.squeeze()).view(
                        generated.shape[0], 1, -1
                    )
                elif "bert" in PRETRAINED.lower():
                    next_token_embed = model.gpt.transformer.wte(next_token.squeeze()).view(
                        generated.shape[0], 1, -1
                    )
                else:
                    import pdb;pdb.set_trace()

                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            import pdb;pdb.set_trace()
            if "galactica" in PRETRAINED.lower():
                end_pos = np.where((np.asarray(output_list) == stop_token_index))[0][0]
                output_text = tokenizer.decode(output_list[1:end_pos])  # Remove end_pos if you want to hallucinate descriptions
            else:
                output_text = sf.decoder(tokenizer.decode(output_list, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(" ", ""))
            generated_list.append(output_text)
    import pdb;pdb.set_trace()
    outs = [compute_tanimoto_similarity(x, labels) for x in generated_list]
    # ref = Chem.MolFromSmiles(labels)
    # outs = []
    # for x in generated_list:
    #     try:
    #          outs.append(TS(Chem.RDKFingerprint(Chem.MolFromSmiles(x)), Chem.RDKFingerprint(ref)))
    #     except:
    #         print("skipped {}".format(x))

    eds = [1 - Levenshtein.distance(labels, x) / max(len(labels), len(x)) for x in generated_list]
    return generated_list, outs, eds  # [0]


def main(
        target=["PKD2"],  # TARDBP"],
        weights_path="checkpoints/coco_prefix-002.pt",
        # prefix_length=30,  # 40,
        beam_search=False,
        num_seqs=10,
        sample=False,
        # test_set=True,
        # permutation=False,
        figsize = (4, 4),
        device=torch.device("cpu")):  # cuda")):

    # config = AutoConfig.from_pretrained("ncfrey/ChemGPT-1.2B")
    # tokenizer = AutoTokenizer.from_pretrained("ncfrey/ChemGPT-1.2B", config=config)

    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix_length', type=int, default=20)
    parser.add_argument('--prefix_length_clip', type=int, default=20)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true', default=True)
    parser.add_argument('--permutation', dest='permutation', action='store_true', default=False)
    parser.add_argument('--test_set', dest='test_set', action='store_true', default=False)
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    args = parser.parse_args()
    prefix_length = args.prefix_length

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED, cache_dir=CACHE_DIR)

    # model = torch.compile(model)
    from modeling_blip_2 import Blip2ForConditionalGeneration
    from configuration_blip_2 import (
        Blip2Config,
    )
    # Initializing BLIP-2 vision, BLIP-2 Q-Former and language model configurations
    text_config = AutoConfig.from_pretrained(PRETRAINED)
    config = Blip2Config(text_config=text_config)
    model = Blip2ForConditionalGeneration(config)
    # model = Blip2ForConditionalGeneration.from_pretrained(weights_path)

    weights = torch.load(weights_path, map_location=CPU)
    keys = [k for k, v in model.named_parameters()]
    both = [k for k in weights.keys() if k in keys]
    leftover =  [k for k in weights.keys() if k not in keys]
    print("Leftover: {}".format(leftover))
    model.load_state_dict(weights, strict=False)
    model = model.eval()
    model = model.to(device)

    preproc = np.load("galactica_cnn_emb_preproc_for_train.npz", allow_pickle=True)
    train_morphology = preproc["train_morphology"]
    mn, mx = train_morphology.min(0), train_morphology.max(0)
    mn = torch.tensor(mn).to(device).float()
    mx = torch.tensor(mx).to(device).float()

    if args.test_set:
        preproc = np.load("galactica_cnn_emb_preproc_for_train.npz", allow_pickle=True)
        morphology = preproc["test_morphology"][:6*8*10]
        selfies = preproc["test_selfies"]
        if args.permutation:
            selfies = selfies[np.random.permutation(len(selfies))]
        selfies = selfies[:6*8*10]
        # morphology = preproc["train_morphology"][:6*8*10]
        # selfies = preproc["train_selfies"][:6*8*10]

        prefixes = torch.tensor(morphology[:num_seqs]).to(device).float()
        labels = selfies[:num_seqs]
    else:
        preproc = np.load("galactica_cnn_emb_preproc_for_test.npz", allow_pickle=True)  # "preproc_for_test.npz")
        orf_morphology = preproc["target"]
        orfs = preproc["orfs"]
        idx = np.where(orfs == target)[0][0]
        prefixes = torch.tensor(orf_morphology[[idx]].repeat(num_seqs, 0)).to(device).float()
        labels = torch.ones(len(prefixes))

    #  Normalize to [-1, 1]
    prefixes = (prefixes - mn) / (mx - mn)
    prefixes = (prefixes - 0.5) / 0.5

    import pdb;pdb.set_trace()
    input_ids = tokenizer.encode(GALACTICA_START) 
    model.generate(image_embeds=prefixes[[0]], input_ids=input_ids)
    preds, outs, eds = [], [], []
    for idx in tqdm(range(num_seqs), total=num_seqs, desc="Sequences"):
        if beam_search:
            pred = generate_beam(model, tokenizer, labels=labels[idx], embed=prefix_embed[[idx]], sample=sample)[0]
        else:
            pred, out, ed = generate2(model, tokenizer, labels=labels[idx], embed=prefix_embed[[idx]], sample=sample, entry_count=1)
        preds.append(pred)
        outs.append(out)
        eds.append(ed)
        # mol = Chem.MolFromSmiles(pred)

        # Generate a 2D depiction of the molecule using RDKit
        # img = Draw.MolToImage(mol, size=figsize)

        # Display the image using matplotlib
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()
        # plt.close("all")

    preds, labels, outs, eds
    if permutation:
        np.savez("perm_test", preds=preds, labels=labels, outs=outs, eds=eds)
    else:
        np.savez("true_test", preds=preds, labels=labels, outs=outs, eds=eds)
    # pred = generate_beam(model, tokenizer, embed=prefix_embed, temperature=1.)
    # pred = generate2(model, tokenizer, embed=prefix_embed)


if __name__ == '__main__':
    main()  # sys.argv[1])

