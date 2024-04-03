from dgllife.utils import load_molecule, mol_to_nearest_neighbor_graph, MolToBigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer, AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from torch_geometric.utils import from_networkx
import pandas as pd
import dgl
from dgl.data.utils import load_graphs, save_graphs
import torch
import networkx as nx
import numpy as np

from biopandas.pdb import PandasPdb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolFromPDBFile
from torch_geometric.utils import negative_sampling

from pyuul import VolumeMaker # the main PyUUL module
from pyuul import utils # the PyUUL utility module
import glob
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
from urllib import request
import urllib.error
from prody import *
import os

from tqdm import tqdm

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# method for creating ct
_repmat = {
    1: ["A", "G", "V"],
    2: ["I", "L", "F", "P"],
    3: ["Y", "M", "T", "S"],
    4: ["H", "N", "Q", "W"],
    5: ["R", "K"],
    6: ["D", "E"],
    7: ["C"],
}

def _Str2Num(proteinsequence):
    """
    translate the amino acid letter into the corresponding class based on the
    given form.
    """
    repmat = {}
    for i in _repmat:
        for j in _repmat[i]:
            repmat[j] = i

    res = proteinsequence
    for i in repmat:
        res = res.replace(i, str(repmat[i]))
    return res

def CalculateConjointTriad(proteinsequence):
    """
    Calculate the conjoint triad features from protein sequence.
    Useage:
    res = CalculateConjointTriad(protein)
    Input: protein is a pure protein sequence.
    Output is a dict form containing all 343 conjoint triad features.
    """
    res = {}
    proteinnum = _Str2Num(proteinsequence)
    for i in range(1, 8):
        for j in range(1, 8):
            for k in range(1, 8):
                temp = str(i) + str(j) + str(k)
                res[temp] = proteinnum.count(temp)
    return torch.tensor(np.array(list(res.values()))).float()

### 

def pdb_to_dynamic_correlation_graph(pdb_id, chain_folder):
    atoms = parsePDB(f'{chain_folder}/{pdb_id}.pdb')
    atoms = atoms.select("calpha")
    res = atoms.getResnames()
    anm = ANM(name=f'{pdb_id}.pdb')
    anm.buildHessian(atoms, cutoff=10, gamma=1, norm=True)

    # """ Kirchhoff matrix """
    # K = anm.getKirchhoff()
    # D = np.diag(np.diag(K) + 1.)

    # """ Contact map """
    # cont = -(K - D)

    """ Mode calculation """
    try:
        anm.calcModes(n_modes=20, zeros=False, turbo=True)
    except Exception as err:
        print(err)
        print("Cannot create ANM model for: ", pdb_id)

    # freqs = []
    # for mode in anm:
    #     freqs.append(math.sqrt(mode.getEigval()))

    corr = calcCrossCorr(anm)

    corr_abs = np.abs(corr)
    corr_abs[corr_abs < 0.5] = 0
    corr_abs[corr_abs > 0.5] = 1

    # diff = cont - corr_abs
    # diff[diff < 0] = -1
    # diff[diff >= 0] = 0

    # """ Adjacency matrix """
        
    # comb = cont + diff # 1: contact edge / -1: correlation edge
    # Adj = np.abs(comb)

    one_hot = amino_acid_one_hot(res)

    u, v = np.nonzero(corr_abs)
    u, v = torch.from_numpy(u), torch.from_numpy(v)
    graph = dgl.to_bidirected(dgl.graph((u, v), num_nodes=len(res)))

    graph.ndata['h'] = one_hot

    return graph

### method for creating residue graph
aa = "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split()
aa_dict = {}

for idx, a in enumerate(aa):
    aa_dict[a] = idx

def amino_acid_one_hot(residues):
    one_hot = torch.from_numpy(np.zeros((len(residues), 20)))
    for idx, res in enumerate(residues):
        one_hot[idx, aa_dict[res]] = 1
    return one_hot

def get_distance_matrix(coords):
    diff_tensor = np.expand_dims(coords, axis=1) - np.expand_dims(coords, axis=0)
    distance_matrix = np.sqrt(np.sum(np.power(diff_tensor, 2), axis=-1))
    return distance_matrix


def pdb_to_res_graph(pdb_id, chain_folder, distance_threshold=6.0, contain_b_factor=False):
    atom_df = PandasPdb().read_pdb(f'{chain_folder}/{pdb_id}.pdb')
    
    atom_df = atom_df.df['ATOM']
    j = atom_df.groupby('residue_number', as_index=False)[['residue_name']].apply(lambda x: x.sum()).sort_values('residue_number')
    j['residue_name'] = j['residue_name'].apply(lambda x: x[:3])
    residues = j['residue_name'].tolist()
    
    one_hot = amino_acid_one_hot(residues)
    residue_df = atom_df.groupby('residue_number', as_index=False)[['x_coord', 'y_coord', 'z_coord', 'b_factor']].mean().sort_values('residue_number')

    coords = residue_df[['x_coord', 'y_coord', 'z_coord']].values
    distance_matrix = get_distance_matrix(coords)
    adj = distance_matrix < distance_threshold
    u, v = np.nonzero(adj)
    u, v = torch.from_numpy(u), torch.from_numpy(v)
    graph = dgl.to_bidirected(dgl.graph((u, v), num_nodes=len(coords)))
#     if contain_b_factor:
#         b_factor = torch.from_numpy(residue_df['b_factor'].values)
#         graph.ndata['b_factor'] = b_factor
    graph.ndata['h'] = one_hot
    
    return graph

def pdb_to_anm_modes(pdb_id, chain_folder):
    atoms = parsePDB(f'{chain_folder}/{pdb_id}.pdb')
    atoms = atoms.select("calpha")

    anm = ANM(name=f'{pdb_id}.pdb')
    anm.buildHessian(atoms, cutoff=10, gamma=1, norm=True)

    """ Mode calculation """
    
    anm.calcModes(n_modes=10, zeros=False, turbo=True)

    freqs = []
    for mode in anm:
        freqs.append(math.sqrt(mode.getEigval()))

    anm_mode = torch.tensor(np.array(freqs)).float()
    
    return anm_mode

#create atom graphs

def pdb_to_atom_bi_graph_atten(pdb_id, chain_folder):
    mol, coords = load_molecule(f'{chain_folder}/{pdb_id}.pdb')
    dg = MolToBigraph(node_featurizer=AttentiveFPAtomFeaturizer(), edge_featurizer = AttentiveFPBondFeaturizer(self_loop=True),add_self_loop=True)
    dg = dg(mol)
    return dg

#create voxels

def voxelize(pdb_id, chain_folder, res=1, dim =5):
    """ Returns the voxelized protein tensor
        @params:
            pdb_fpath: file path of pdb
            res: side in Angstrum of a voxel. The lower this value is the higher the resolution of the final representation will be
            dim: maximum distance in number of voxels for which the contribution to occupancy is taken into consideration. Every atom that is farer than cubes_around_atoms_dim voxels from the center of a voxel does no give any contribution to the relative voxel occupancy
        @return:
            voxelized protein tensor
    """

    coords, atname = utils.parsePDB(f'{chain_folder}/{pdb_id}.pdb') # get coordinates and atom names
    atoms_channel = utils.atomlistToChannels(atname) # calculates the corresponding channel of each atom
    radius = utils.atomlistToRadius(atname) # calculates the radius of each atom

    device = "cuda:0" # runs the volumes on CPU
    VoxelsObject = VolumeMaker.Voxels(device=device)

    coords = coords.to(device)
    radius = radius.to(device)
    atoms_channel = atoms_channel.to(device)

    VoxelRepresentation = VoxelsObject(coords, radius, atoms_channel,resolution=res,cubes_around_atoms_dim=dim).to_dense().to('cpu')
    VoxelRepresentation = VoxelRepresentation.sum(1).numpy()

    VoxelRepresentation = torch.tensor(np.round(VoxelRepresentation, 4))

    # VoxelRepresentation = torch.where(VoxelRepresentation>0.9, 1.0, 0.0)
    VoxelRepresentation = VoxelRepresentation[None, :]

    VoxelsObject = VoxelsObject.to('cpu')
    coords = coords.to('cpu')
    radius = radius.to('cpu')
    atoms_channel = atoms_channel.to('cpu')

    return VoxelRepresentation

def remove_empty_voxels(ten):
    """ Returns voxel tensor with extra padding removed
        @params:
            ten: voxel tensor
        @return:
            tensor with padding removed
    """

    indices = torch.nonzero(ten[0,0], as_tuple=True)

    min_2 = min(indices[0])
    max_2 = max(indices[0])
    min_3 = min(indices[1])
    max_3 = max(indices[1])
    min_4 = min(indices[2])
    max_4 = max(indices[2])

    return ten[:,:,min_2:max_2+1,min_3:max_3+1,min_4:max_4+1]

def min_res_vox(pdb_id):
    """ Returns the smallest possible voxelized tensor of given pdb file name within dimensions of 32x32x32
        @params:
            pdb_fname: file path of pdb
        @return:
            voxelized protein tensor
    """
    i = 1
    ten = voxelize(pdb_id, i)
    ten = remove_empty_voxels(ten)
    
    while (ten.shape[2] > 32) or (ten.shape[3] > 32) or (ten.shape[4] > 32):
        if i == 6:
            raise Exception("Resolution cannot be lowered further")
        i += 1
        ten = voxelize(pdb_id, i)
        ten = remove_empty_voxels(ten)
    
    ten = F.pad(ten, [0, 32 - ten.size(4), 0, 32 - ten.size(3), 0, 32 - ten.size(2)]).view(-1).double()
        #plot_voxel((ten[0].sum(0) > 0.9).detach().numpy())
    
    vox = torch.where(ten > 0.75, 1.0, 0.0).float()
    return vox

##for llm embedding

#@title Load encoder-part of ProtT5 in half-precision. { display-mode: "form" }
# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50 in half-precision)
transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
print("Loading: {}".format(transformer_link))
model = T5EncoderModel.from_pretrained(transformer_link)
model.full() if device=='cpu' else model.half() # only cast to full-precision if no GPU is available
model = model.to(device)
model = model.eval()
tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False )

def seq_to_llm(seq):
    sequence_examples = [seq]
    # this will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)

    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids,attention_mask=attention_mask)

    llm = embedding_repr.last_hidden_state.mean(dim=1)
    return torch.tensor(llm[0]).float()


def download_protein_representations(pdb_chains, seq_dict, chain_folder):

    for pdb_id in tqdm(pdb_chains, desc="Creating protein representations"):

        try:
            if not os.path.exists(f'ct/{pdb_id}.pt'):
                seq = seq_dict[pdb_id]
                # seq = uni_to_seq[pdb_id]
                if not (len(seq) < 50 or "U" in seq or "X" in seq):
                    ct = CalculateConjointTriad(seq)
                    torch.save(ct, f'ct/{pdb_id}.pt')

        except Exception as e:
            print(e)
            print(f'Seq issue with {pdb_id}')
            continue

        try:

            # if not os.path.exists(f'{graph_folder}/{pdb_id}.pt'):
            dg = pdb_to_atom_bi_graph_atten(pdb_id, chain_folder)

            if dg.num_edges() != 0:
                save_graphs(dg, f'graphs/{pdb_id}.pt')

        except Exception as e:
            print(e)
            print(f'Atomic graph issue with {pdb_id}')
            continue

        try:
            if not os.path.exists(f'res_dgs/{pdb_id}.pt'):
                dg = pdb_to_res_graph(pdb_id, chain_folder)

                if dg.num_edges() != 0:
                    torch.save(dg, f'res_dgs/{pdb_id}.pt')


        except Exception as e:
            print(e)
            print(f'Residue graph issue with {pdb_id}')
            continue

        try:
            # if not os.path.exists(f'{dyn_graph_folder}/{pdb_id}.pt'):
            dg = pdb_to_dynamic_correlation_graph(pdb_id, chain_folder)
            save_graphs(dg, f'dyn_graphs/{pdb_id}.pt')

        except Exception as e:
            print(f'GNM issue with {pdb_id}')
            print(e)


        try:
            if not os.path.exists(f'h_gnm/{pdb_id}.pt'):
                anm = pdb_to_anm_modes(pdb_id, chain_folder)
                torch.save(anm, f'h_gnm/{pdb_id}.pt')
        except Exception as e:
            print(f'ANM issue with {pdb_id}')
            print(e)

        try:
            ten = min_res_vox(pdb_id, chain_folder)
            torch.save(ten,f'voxel/{pdb_id}.pt')
        except Exception as e:
            print(e)
            print(f'Voxel issue with {pdb_id}')

        try:
            if not os.path.exists(f'ct/{pdb_id}.pt'):
                seq = seq_dict[pdb_id]
                # seq = uni_to_seq[pdb_id]
                if not (len(seq) < 50 or "U" in seq or "X" in seq):
                    llm = seq_to_llm(seq)
                    torch.save(llm, f'llm/{pdb_id}.pt')

        except Exception as e:
            print(e)
            print(f'LLM issue with {pdb_id}')
            continue