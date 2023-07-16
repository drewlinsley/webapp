import numpy as np
import pandas as pd
import selfies as sf
from tqdm import tqdm
from rdkit import Chem
from molvs import standardize_smiles
from openbabel import openbabel
from pubchempy import get_compounds
from joblib import Parallel, delayed
from openeye.oechem import OEGraphMol, OEParseInChI, OECreateCanSmiString


def smiles_from_inchi_pubchem(x, tries=10):
    t = 0
    while t < tries:
        try:
            comp = get_compounds(x, "inchi")
            t = tries + 10
        except:
            pass  # Trying again
        t += 1
    if t == tries:
        return "nan"  # Didnt work
    check = hasattr(comp[0], "isometric_smiles")
    if check:
        return comp[0].isometric_smiles
    else:
        return "nan"


def inchi_to_smiles(inchi_string, pc):
    mol = Chem.inchi.MolFromInchi(inchi_string)
    if mol is not None:
        mol = Chem.MolToSmiles(mol, kekuleSmiles=False, allBondsExplicit=True)
        if mol in pc:
            return mol
        else:
            return "nan"
    else:
        return "nan"


def open_inchi_to_smiles(inchi_string, pc):
    mol = OEGraphMol()
    if not OEParseInChI(mol, inchi_string):
        return "nan"
    smiles_string = OECreateCanSmiString(mol)
    return smiles_string


def get_smiles_from_inchi(inchi, pc):
    """Convert smiles to inchi via openbabel.

    Thank you openai chatbot!"""
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("inchi", "can")  # smi

    mol = openbabel.OBMol()
    obConversion.ReadString(mol, inchi)

    smiles = obConversion.WriteString(mol)
    import pdb;pdb.set_trace()
    smiles = smiles.split("\t")[0]
    if smiles in pc:
        return smiles
    else:
        return "nan"


# PRETRAINED = "ncfrey/ChemGPT-1.2B"
PRETRAINED = "ncfrey"
PRETRAINED = "galactica"
pca = False
average = False
split = "hold_out_mol"
if average:
    hold_out = 500
else:
    hold_out = 50
# data = np.load("../linear_models/normalized_data-13.npz")
# fn = "../linear_models/normalized_data-13.npz"
# fn = "../linear_models/controlled_data-13.npz"
# fn = "../linear_models/no_reduce_normalized_data-13.npz"
fn = "../datasets/dl_compound_data-84.parquet"
ofn = "../datasets/dl_orf_data-84.parquet"
# fn = "../linear_models/no_reduce_controlled_data-13.npz"
data = pd.read_parquet(fn)
orf_data = pd.read_parquet(ofn)

#     np.savez("batch_normalized_prepped_data.npz".format(target), inchi=inchi, comp_data=comp_data, orf_data=orf_data, orfs=orfs)
rna_mus, agp_mus, dna_mus, er_mus, mito_mus = {}, {}, {}, {}, {}
rna_sds, agp_sds, dna_sds, er_sds, mito_sds = {}, {}, {}, {}, {}

batches = data.Metadata_Batch.unique()
inchis, rna, agp, dna, er, mito = [], [], [], [], [], []
for batch in batches:
    idx = batch == data.Metadata_Batch
    di = data[idx]
    inchi = di.Metadata_InChI.values.astype(str)
    inchis.append(inchi)
    rna_emb = np.stack(di.rna_emb.values, 0)
    agp_emb = np.stack(di.agp_emb.values, 0)
    dna_emb = np.stack(di.dna_emb.values, 0)
    er_emb = np.stack(di.er_emb.values, 0)
    mito_emb = np.stack(di.mito_emb.values, 0)

    rna_mu, rna_sd = rna_emb.mean(0), rna_emb.std(0)
    agp_mu, agp_sd = agp_emb.mean(0), agp_emb.std(0)
    dna_mu, dna_sd = dna_emb.mean(0), dna_emb.std(0)
    er_mu, er_sd = er_emb.mean(0), er_emb.std(0)
    mito_mu, mito_sd = mito_emb.mean(0), mito_emb.std(0)

    rna_mus[batch], rna_sds[batch] = rna_mu, rna_sd
    agp_mus[batch], agp_sds[batch] = agp_mu, agp_sd
    dna_mus[batch], dna_sds[batch] = dna_mu, dna_sd
    er_mus[batch], er_sds[batch] = er_mu, er_sd
    mito_mus[batch], mito_sds[batch] = mito_mu, mito_sd

    rna_emb = (rna_emb - rna_mu) / rna_sd
    agp_emb = (agp_emb - agp_mu) / agp_sd
    dna_emb = (dna_emb - dna_mu) / dna_sd
    er_emb = (er_emb - er_mu) / er_sd
    mito_emb = (mito_emb - mito_mu) / mito_sd

    rna.append(rna_emb)
    agp.append(agp_emb)
    dna.append(dna_emb)
    er.append(er_emb)
    mito.append(mito_emb)
inchi = np.concatenate(inchis)
rna_emb = np.concatenate(rna, 0)
agp_emb = np.concatenate(agp, 0)
dna_emb = np.concatenate(dna, 0)
er_emb = np.concatenate(er, 0)
mito_emb = np.concatenate(mito, 0)

morphology = np.stack((rna_emb, agp_emb, dna_emb, er_emb, mito_emb), 1).astype(np.float32)
mask = []
for i in inchi:
    if i is None:
        mask.append(False)
    else:
        mask.append(True)
mask = np.asarray(mask)
inchi = inchi[mask]
morphology = morphology[mask]

if not average:
    # Remove compounds with imbalanced entries
    max_comp = 100
    un, co = np.unique(inchi, return_counts=True)
    remove = un[co > max_comp]  # Remove compounds with > 100 entries
    # remove = un[co < max_comp]  # Remove compounds with > 100 entries
    mask = []
    for r in inchi:
        if r in remove:
            mask.append(False)
        else:
            mask.append(True)
    mask = np.asarray(mask)
    inchi = inchi[mask]
    morphology = morphology[mask]
else:
    # Average morphology
    un = np.unique(inchi)
    res_morph = morphology.reshape(-1, 5 * 1280)
    morphology = pd.DataFrame(res_morph)
    inchi = pd.DataFrame(inchi, columns=["inchi"])
    both = pd.concat((morphology, inchi), 1)
    both = both.groupby("inchi").mean()
    morphology = both.values.reshape(-1, 5, 1280)
    inchi = both.reset_index().inchi.values

# Convert to SMILES
smile_conv = pd.read_csv("pubchem_smiles.csv")
smiles = []
cache = {}
for inch in tqdm(inchi, total=len(inchi)):
    check = smile_conv['0'] == inch
    if inch in cache:
        smiles.append(cache[inch])
    else:
        if check.sum():
            smile = smile_conv[check]['1'].values[0]
        else:
            smile = "nan"
        smiles.append(smile)
        cache[inch] = smile
selfies = np.asarray(smiles)

# selfies = np.asarray([inchi_to_smiles(x, pubchem) for x in tqdm(inchi, total=len(inchi))])
# selfies = np.asarray([open_inchi_to_smiles(x, pubchem) for x in tqdm(inchi, total=len(inchi))])
# selfies = np.asarray([get_smiles_from_inchi(x, pubchem) for x in tqdm(inchi, total=len(inchi))])

# smiles = np.asarray([smiles_from_inchi_pubchem(x) for x in tqdm(inchi, total=len(inchi))])
# selfies = np.asarray(Parallel(n_jobs=-1)(delayed(smiles_from_inchi_pubchem)(x) for x in tqdm(inchi, total=len(inchi))))

# Load pubchem into memory
mask = selfies != "nan"
selfies = selfies[mask]
morphology = morphology[mask]
inchi = inchi[mask]
print("Using {} examples".format(len(selfies)))

if split == "hold_out_mol":
    un, inv = np.unique(selfies, return_inverse=True)
    test_mask = np.zeros(len(selfies))
    for u in range(hold_out):  # len(un)):
        idx = u == inv
        test_mask[idx] = 1
else:
    # Split into train/test
    test = {}
    for idx, (s, i, m) in enumerate(zip(selfies, inchi, morphology)):
        if i not in test:
            test[i] = idx
    idxs = np.asarray([v for v in test.values()])
    test_mask = np.zeros(len(selfies))
    test_mask[idxs] = 1

test_selfies = selfies[test_mask == 1]
test_morphology = morphology[test_mask == 1]
train_selfies = selfies[test_mask == 0]
train_morphology = morphology[test_mask == 0]

# Now load orfs
orf_data = orf_data[orf_data.Metadata_pert_type == "trt"]
orf_data = orf_data[orf_data.Metadata_Symbol != "nan"]
batch = orf_data.Metadata_Batch.unique()[0]
orfs = orf_data.Metadata_Symbol.values
orna_emb = np.stack(orf_data.rna_emb.values, 0)
oagp_emb = np.stack(orf_data.agp_emb.values, 0)
odna_emb = np.stack(orf_data.dna_emb.values, 0)
oer_emb = np.stack(orf_data.er_emb.values, 0)
omito_emb = np.stack(orf_data.mito_emb.values, 0)

orna_emb = (orna_emb - rna_mus[batch]) / rna_sds[batch]
oagp_emb = (oagp_emb - agp_mus[batch]) / agp_sds[batch]
odna_emb = (odna_emb - dna_mus[batch]) / dna_sds[batch]
oer_emb = (oer_emb - er_mus[batch]) / er_sds[batch]
omito_emb = (omito_emb - mito_mus[batch]) / mito_sds[batch]
target = np.stack((orna_emb, oagp_emb, odna_emb, oer_emb, omito_emb), 1).astype(np.float32)

mm, mx = 0, 1
# mm, mx = train_morphology.mean((0, 2), keepdims=True), train_morphology.std((0, 2), keepdims=True)
# train_morphology = (train_morphology - mm) / mx
# test_morphology = (test_morphology - mm) / mx
# target = (target - mm) / mx

prefix = "cnn_emb"
prefix = "{}_{}".format(PRETRAINED, prefix)
np.savez("{}_preproc_for_train".format(prefix), train_morphology=train_morphology, test_morphology=test_morphology, train_selfies=train_selfies, test_selfies=test_selfies)
np.savez("{}_preproc_for_test".format(prefix), mm=mm, mx=mx, thresh=-1, target=target, orfs=orfs)

