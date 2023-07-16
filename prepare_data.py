import numpy as np
import pandas as pd
import selfies as sf
from tqdm import tqdm
from rdkit import Chem
# from molvs import standardize_smiles
# from openbabel import openbabel
from pubchempy import get_compounds


def smiles_from_inchi_pubchem(x):
    comp = get_compounds(x, "inchi")
    if len(comp):
        return comp.isometric_smiles
    else:
        return "nan"


def inchi_to_smiles(inchi_string):
    mol = Chem.inchi.MolFromInchi(inchi_string)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    else:
        return "nan"


def get_smiles_from_inchi(inchi):
    """Convert smiles to inchi via openbabel.

    Thank you openai chatbot!"""
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("inchi", "smi")

    mol = openbabel.OBMol()
    obConversion.ReadString(mol, inchi)

    smiles = obConversion.WriteString(mol)
    return smiles.split("\t")[0]


# PRETRAINED = "ncfrey/ChemGPT-1.2B"
PRETRAINED = "ncfrey"
PRETRAINED = "galactica"
PRETRAINED = "smiles-gpt"
pca = False
average = False
split = "ORF_reuse"  # Reuse existing ORF indices
if average:
    hold_out = 500
else:
    hold_out = 500
# data = np.load("../linear_models/normalized_data-13.npz")
# fn = "../linear_models/batch_data-18.npz"
fn = "../linear_models/batch_data-24.npz"
data = np.load(fn, allow_pickle=True)
assert "source" in data.keys(), "Need a source index so that we can predict on the ORF-generating source."
assert "orf_source" in data.keys(), "Need the ID for the ORF-generating source"
inchi = data["compounds"].squeeze()
inchi_name = data["inchis"].squeeze()
morphology = data["comp_data"].astype(np.float32)  # [:, None].astype(np.float32)
source = data["source"].squeeze()
well = data["well"].squeeze()  # Load wells so we can check for a well bias
smiles = data["smiles"].squeeze()
selfies = data["selfies"].squeeze()
orf_source = data["orf_source"][0]
target = data["orf_data"].squeeze()
orfs = data["orfs"].squeeze()
orf_well = data["orf_well"].squeeze()
orf_inds = np.load("../linear_models/test_meta.npz")
crisprs = data["crisprs"].squeeze()
crispr_well = data["crispr_well"].squeeze()
crispr_target = data["crispr_data"].squeeze()

reuse = orf_inds["test_X_idx"]
reuse_idx = np.zeros_like(well)
reuse_idx[reuse] = 1  # For re-using the encoder test set

mask = morphology.squeeze().sum(1) != 0
morphology = morphology[mask]
inchi = inchi[mask]
inchi_name = inchi_name[mask]
source = source[mask]
well = well[mask]
reuse_idx = reuse_idx[mask]

"""
# Remove empty inchis
im = inchi != "empty"
morphology = morphology[im]
inchi = inchi[im]
source = source[im]
well = well[im]
reuse_idx = reuse_idx[im]
"""

pre_size = len(inchi)
if 1:
    pass
elif not average:
    # Remove compounds with imbalanced entries
    un, co = np.unique(inchi, return_counts=True)
    remove = un[co > 100000]  # Remove compounds with > 100 entries
    mask = []
    for r in inchi:
        if r in remove:
            mask.append(False)
        else:
            mask.append(True)
    mask = np.asarray(mask)
    inchi = inchi[mask]
    inchi_name = inchi_name[mask]
    morphology = morphology[mask]
    source = source[mask]
    well = well[mask]
    reuse_idx = reuse_idx[mask]
else:
    # Average morphology
    un = np.unique(inchi)
    import pandas as pd
    morphology = pd.DataFrame(morphology)
    inchi = pd.DataFrame(inchi, columns=["inchi"])
    both = pd.concat((morphology, inchi), 1)
    both = both.groupby("inchi").mean()
    morphology = both.values 
    inchi = both.reset_index().inchi.values
post_size = len(inchi)
print("{} compounds remaining ({} removed)".format(post_size, pre_size - post_size))

# Convert to SMILES
# smiles = np.asarray([get_smiles_from_inchi(x) for x in tqdm(inchi, total=len(inchi))])
# smiles = np.asarray([inchi_to_smiles(x) for x in tqdm(inchi, total=len(inchi))])
# smiles = np.asarray([smiles_from_inchi_pubchem(x) for x in tqdm(inchi, total=len(inchi))])
# mask = smiles != "nan"
# smiles = smiles[mask]
# inchi = inchi[mask]
# morphology = morphology[mask]
# smiles = np.asarray([standardize_smiles(x) for x in tqdm(smiles, total=len(smiles), desc="Normalizing smiles")])

# Convert to SMILES
"""
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

# Load pubchem into memory
mask = selfies != "nan"
selfies = selfies[mask]
morphology = morphology[mask]
inchi = inchi[mask]
source = source[mask]
well = well[mask]
reuse_idx = reuse_idx[mask]
print("Using {} examples".format(len(selfies)))

if "galactica"  not in PRETRAINED:
    # Convert to selfies
    fselfies, keeps = [], []
    for s in tqdm(selfies, total=len(selfies), desc="Converting smiles to selfies"):
        try:
            enc_sf = sf.encoder(s)
            fselfies.append(enc_sf)
            keeps.append(True)
        except:
            keeps.append(False)
    keeps = np.asarray(keeps)
    morphology = morphology[keeps]
    inchi = inchi[keeps]
    source = source[mask]
    well = well[mask]
    reuse_idx = reuse_idx[mask]
else:
    fselfies = selfies
selfies = np.asarray(fselfies)
"""
# Truncate some of the crazier rows and normalize to [0, 1]
thresh = 1000000
"""
if "normalized" in fn:
    thresh = 50
else:
    thresh = 1000000
if thresh:
    mask = (np.abs(morphology.squeeze()) > thresh).sum(1) == 0
else:
    mask = np.ones((len(morphology)))
print("Keeping {}/{} (removing {})".format(mask.sum(), len(mask), len(mask) - mask.sum()))

selfies = selfies[mask]
inchi = inchi[mask]
morphology = morphology[mask]
source = source[mask]
well = well[mask]
"""

# Remove 0 cols
zer = morphology.sum(0) != 0
morphology = morphology[..., zer]

if split == "hold_out_mol":
    un, inv = np.unique(inchi, return_inverse=True)
    test_mask = np.zeros(len(selfies))
    for u in range(hold_out):  # len(un)):
        idx = u == inv
        test_mask[idx] = 1
elif split == "ORF_source":
    idx = source == orf_source
    ids = np.where(idx)[0][:hold_out]
    test_mask = np.zeros(len(selfies))
    test_mask[ids] = 1
elif split == "ORF_reuse":
    test_mask = np.zeros(len(selfies)).astype(int)
    test_mask[reuse_idx == 1] = 1
else:
    # Split into train/test
    test = {}
    for idx, (s, i, m) in enumerate(zip(selfies, inchi, morphology)):
        if i not in test:
            test[i] = idx
    idxs = np.asarray([v for v in test.values()])
    test_mask = np.zeros(len(selfies))
    test_mask[idxs] = 1
print("Using {} test indices".format(test_mask.sum()))
test_selfies = selfies[test_mask == 1]
test_smiles = smiles[test_mask == 1]
test_inchis = inchi_name[test_mask == 1]
test_morphology = morphology[test_mask == 1]
test_source = source[test_mask == 1]
test_well = well[test_mask == 1]
train_selfies = selfies[test_mask == 0]
train_smiles = smiles[test_mask == 0]
train_inchis = inchi_name[test_mask == 0]
train_morphology = morphology[test_mask == 0]
train_source = source[test_mask == 0]
train_well = well[test_mask == 0]

"""
# Aggregate test data
sf, mo, so, we, sm = [], [], [], [], []
uni_slf = np.unique(test_selfies)
for s in uni_slf:
    idx = test_selfies == s
    wl = test_well[idx]
    uw = np.unique(wl)
    for w in uw:
        cidx = np.logical_and(idx, test_well == w)
        sf.append(test_selfies[cidx][0])
        mo.append(np.median(test_morphology[cidx], 0))
        so.append(test_source[cidx][0])
        we.append(test_well[cidx][0])
        sm.append(test_smiles[cidx][0])
    # idx = s == test_selfies
    # sf.append(test_selfies[idx])
    # mo.append(test_morphology[idx].mean(0))
    # so.append(test_source[idx])
    # we.append(test_well[idx])
test_selfies = np.asarray(sf)
test_smiles = np.asarray(sm)
test_morphology = np.stack(mo, 0)
test_source = np.asarray(so)
test_well = np.asarray(we)
"""

# Prepare ORF data
mask = target.squeeze().sum(1) != 0
target = target[mask]
orfs = orfs[mask]
orf_well = orf_well[mask]
mask = orfs != "nan"
target = target[mask]
orfs = orfs[mask]
orf_well = orf_well[mask]

# Prepare CRISPR data
mask = crispr_target.squeeze().sum(1) != 0
crispr_target = crispr_target[mask]
crisprs = crisprs[mask]
crispr_well = crispr_well[mask]
mask = crisprs != "nan"
crispr_target = crispr_target[mask]
crisprs = crisprs[mask]
crispr_well = crispr_well[mask]

mm, mx = 0, 1

if pca:
    from sklearn.decomposition import PCA
    # morphology = morphology.squeeze(1)
    # target = target.squeeze(1)
    pca = PCA(n_components=1024, whiten=True).fit(train_morphology)
    train_morphology = pca.transform(train_morphology)  # [:, None]
    test_morphology = pca.transform(test_morphology)  # [:, None]
    target = pca.transform(target)  # [:, None]

if "ncfrey" in PRETRAINED:
    train_mols = train_selfies
    test_mols = test_selfies
else:
    train_mols = train_smiles
    test_mols = test_smiles

if "normalized" in fn:
    prefix = "normalized"
else:
    prefix = "controlled"
if average:
    prefix = "average_{}".format(prefix)
prefix = "{}_{}".format(PRETRAINED, prefix)
np.savez("{}_preproc_for_train".format(prefix), train_morphology=train_morphology, test_morphology=test_morphology, train_mols=train_mols, test_mols=test_mols, train_selfies=train_selfies, test_selfies=test_selfies, train_inchis=train_inchis, test_inchis=test_inchis)
np.savez("{}_preproc_for_test".format(prefix), mm=mm, mx=mx, thresh=thresh, target=target, orfs=orfs, crispr_target=crispr_target, crisprs=crisprs)

