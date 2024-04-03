# %%
import pandas as pd
from os.path import exists
import numpy as np
from six.moves.urllib.request import urlopen
import requests
import json
import os
import gc
import subprocess
import time
from pdb_to_representations import download_protein_representations 
from dgl.data.utils import load_graphs, save_graphs
from sklearn.model_selection import KFold
import argparse
import torch
import random
from tqdm import tqdm
import networkx as nx
from itertools import combinations

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

set_seed(88)

cuda0 = torch.device('cuda:0')
device = torch.device("cpu")

parser = argparse.ArgumentParser(description='create positive and negative interaction datasets')
parser.add_argument('-ID_mapping', default='../data/string_phy_human_mapping.tsv', type=str, help='path to uniprot mapping file')
parser.add_argument('-dict_data', default='../data/string_phy_human_pdb_chains_dict.pt', type=str, help='path to pdb to chains information dictionary')
parser.add_argument('-pos_int_data', default='../data/string_phy_human.txt', type=str, help='path to postive interaction dataset')
parser.add_argument('-original', action='store_true', help='A boolean argument to specify to not use proteins used in the multiconformation dataset')
parser.add_argument('-STRING', action='store_true', help='A boolean argument to specify if the PPI is from STRING database, if so only interactions with greater than 70% confidence are used')
parser.add_argument('-eval_fold', action='store_true', help='A boolean argument to specify to create 5 fold cross validation for performance evaluation')
parser.add_argument('-prefix', required=True, type=str, help='prefix to save all output files to')
parser.add_argument('-pdb_chains_dname', default='../pdb_chains', type=str, help='path to pdb of single chains directory')
parser.add_argument('-hydrated_pdbs_dname', default='../pdbs', type=str, help='path to hydrated pdbs directory')
static_args = parser.parse_args()
intID_uni_seq_pdbs_fname = static_args.ID_mapping#"idmapping_2023_07_03.tsv" ##ARG
pdb_chains_dict_path = static_args.dict_data
creating_original = static_args.original
pdb_chains_dname = static_args.pdb_chains_dname
hydrated_pdbs_dname = static_args.hydrated_pdbs_dname
positive_network_fname = static_args.pos_int_data#"string_phy_homo_sapien.txt" ##ARG
STRING_DATABASE = static_args.STRING
prefix = static_args.prefix
eval_fold = static_args.eval_fold

# %%
pdb_chains_dict = torch.load(pdb_chains_dict_path) # "../StructPPI/string_hum_pdb_chains_dict.pt" more complete 
# %%
uni_pdbs = {}

#uniprot ids of multi conformation set from the PRISM predictions to remove
# multi_chains = {"1J6ZA":0,"1P7QA":0,"1NBFABE":0,"1NBFCD":0,"1B17A":0,"2HTHB":0,"1P7QB":0,"2G45BE":0,"2HTHA":0,"4UN2A":0,"1FINAC":0,"1ATNA":0,"4UN2B":0,"2G45AD":0,"1B17B":0,"1G0XA":0,"1XD3AC":0,"1ATND":0,"1XD3BD":0,"1HCLA":0,"1FINBD":0}
multi_unis = {'P48510':0,'P61769':0,'P04439':0,'P20248':0,'P68135':0,'P00639':0,'P0CG48':0,'P45974':0,'P15374':0,'Q93009':0,'P24941':0,'P42771':0,'Q86VN1':0,'Q00534':0,'Q8NHL6':0}

print("Creating mapping from uniprot to PDB IDs")

tqdm.pandas()

uniprot_df = pd.read_csv(intID_uni_seq_pdbs_fname, sep="\t").dropna()

if creating_original:
    def create_mapping(row):
        if row["Entry"] not in multi_unis:
            cur_pdbs = row["PDB"][:-1].split(";")

            uni_pdbs[row["Entry"]] = cur_pdbs
        return row
else:
    def create_mapping(row):
        cur_pdbs = row["PDB"][:-1].split(";")

        uni_pdbs[row["Entry"]] = cur_pdbs
        return row

uniprot_df.progress_apply(create_mapping,axis=1)

print("Filtering mapping to PDB IDs that contain proteins interacting or physically bounded")

def assign_pdbs(r):
    try:
        cur_pdbs = uni_pdbs[r.Entry]
        cur_pdbs = [pdb_id for pdb_id in cur_pdbs if (pdb_id in pdb_chains_dict)]
        if len(cur_pdbs) == 0:
            return np.nan
        return cur_pdbs
            
    except KeyError:
        return np.nan
    

uniprot_df["PDB"] = uniprot_df.progress_apply(assign_pdbs, axis =1)
uniprot_df = uniprot_df.dropna()

string_uni_dict = dict(zip(uniprot_df['From'], uniprot_df['Entry']))

# %%

#collect pdb chain ids from the overall graph and positive links

positive_links = []
negative_links = []

seq_dict = {}

#then convert net to the format of uniprot1, unprot2 using uniport mapping tool in a file for both postive and negative dataset


pos_net_df = pd.read_csv(positive_network_fname,sep=" ").dropna()

#remove the reverse order duplicate
pos_net_df = pos_net_df.loc[pd.DataFrame(np.sort(pos_net_df[['protein1','protein2']],1),index=pos_net_df.index).drop_duplicates(keep='first').index]

print("Total unique positive interactions: ", pos_net_df.shape[0])

print("Mapping graph ids from positive links dataset to PDB IDs")

if STRING_DATABASE:
    pos_net_df = pos_net_df[pos_net_df["combined_score"]>=700]

def map_string(r, col_name):
    try:
        cur_uni = string_uni_dict[r[col_name]]
        return pd.Series([cur_uni, uni_pdbs[cur_uni]])
    except KeyError:
        return pd.Series([np.nan,np.nan])

pos_net_df[["uni1", "pdbs1"]] = pos_net_df.progress_apply(lambda r: map_string(r,"protein1"), axis=1)
pos_net_df[["uni2", "pdbs2"]] = pos_net_df.progress_apply(lambda r:map_string(r,"protein2"), axis=1)

#remove the reverse order duplicate
pos_net_df = pos_net_df.dropna()

print("Total unique positive interactions after uniprot mapping: ", pos_net_df.shape[0])

G = nx.from_pandas_edgelist(pos_net_df, 'uni1','uni2')

all_edges = set(combinations(G.nodes(), 2))
connected_edges = set(nx.edges(G))
non_interacting_edges = all_edges - connected_edges

non_interacting_distances = []
for neg_edge in tqdm(non_interacting_edges, desc="Calculating shortest distance between non-interacting edges"):
    try:
        distance = nx.shortest_path_length(G, neg_edge[0], neg_edge[1])
        if distance >= 5:
            non_interacting_distances.append((neg_edge[0], neg_edge[1], distance))
    except nx.NetworkXNoPath:
        non_interacting_distances.append((neg_edge[0], neg_edge[1], 0))

# Create a new DataFrame to store the results
negatome_df = pd.DataFrame(non_interacting_distances, columns=['uni1','uni2', 'distance'])

#remove the reverse order duplicate
negatome_df = negatome_df.loc[pd.DataFrame(np.sort(negatome_df[['uni1','uni2']],1),index=negatome_df.index).drop_duplicates(keep='first').index]

print("Total unique negative interactions: ", negatome_df.shape[0])

del non_interacting_distances
del connected_edges
del all_edges
del G
gc.collect()

#create mappings from graph id to pdb_chain id and reverse
graphid_to_pdb_chains_dict = {}
pdb_chains_to_graphid_dict = {}

def find_chains(pdb_id, uni1, uni2):

    content = pdb_chains_dict[pdb_id]

    chains_1 = ""
    command_chains_1 = ""

    for c in sorted(list(set([entity['chain_id'] for entity in content[uni1]['mappings']]))):
        chains_1 += c
        command_chains_1 += c+','
    
    command_chains_1 = command_chains_1[:-1]
    pdb_chain_1 =  pdb_id + chains_1

    if not os.path.exists(f'{pdb_chains_dname}/{pdb_chain_1}.pdb'):
        # raise Exception('selchain does not work')
        com = os.system(f"pdb_selchain -{command_chains_1} {hydrated_pdbs_dname}/{pdb_id}.pdb | pdb_tidy > {pdb_chains_dname}/{pdb_chain_1}.pdb")
        if not os.path.exists(f'{pdb_chains_dname}/{pdb_chain_1}.pdb'):
            # print('selchain does not work for ', pdb_id, "when extracting", command_chains_1)
            raise Exception('selchain does not work')

    chains_2 = ""
    command_chains_2 = ""
    
    for c in sorted(list(set([entity['chain_id'] for entity in content[uni2]['mappings']]))):
        chains_2 += c
        command_chains_2 += c+','
    
    command_chains_2 = command_chains_2[:-1]
    pdb_chain_2 =  pdb_id + chains_2

    if not os.path.exists(f'{pdb_chains_dname}/{pdb_chain_2}.pdb'):
        # raise Exception('selchain does not work')
        com = os.system(f"pdb_selchain -{command_chains_2} {hydrated_pdbs_dname}/{pdb_id}.pdb | pdb_tidy > {pdb_chains_dname}/{pdb_chain_2}.pdb")
        if not os.path.exists(f'{pdb_chains_dname}/{pdb_chain_2}.pdb'):
            # print('selchain does not work for ', pdb_id, "when extracting", command_chains_2)
            raise Exception('selchain does not work')

    graphid_to_pdb_chains_dict[uni1] = pdb_chain_1
    graphid_to_pdb_chains_dict[uni2] = pdb_chain_2

    pdb_chains_to_graphid_dict[pdb_chain_1] = uni1
    pdb_chains_to_graphid_dict[pdb_chain_2] = uni2
    
    seq_dict[pdb_chain_1] = uniprot_df[uniprot_df.Entry==uni1].Sequence.values[0]
    seq_dict[pdb_chain_2] = uniprot_df[uniprot_df.Entry==uni2].Sequence.values[0]

    return pdb_chain_1, pdb_chain_2

print("Mapping graph ids from positive links dataset to specific PDB chain IDs")

def map_chains(row):

    #fetch list of pdbs for each id

    com_pdbs = sorted(list(set(row.pdbs1).intersection(set(row.pdbs2))))
    
    if len(com_pdbs) > 0:
        
        for pdb in com_pdbs:
            try:
                pdbA, pdbB = find_chains(pdb, row["uni1"], row["uni2"])
                
                curr_int = (pdbA,pdbB)
                positive_links.append(curr_int)
                
                return pd.Series([pdbA,pdbB])
                
            except Exception as e:
                # print(e)
                continue

    return pd.Series([np.nan,np.nan])
    
pos_net_df[['chain1','chain2']] = pos_net_df.progress_apply(map_chains,axis=1)
pos_net_df = pos_net_df.dropna()

print("Total unique positive interactions after finding chains: ", pos_net_df.shape[0])

torch.save(seq_dict, f"../data/{prefix}_seq_dict.pt")
print("Chain to sequence dictionary saved to ",  f"../data/{prefix}_seq_dict.pt")

def map_chains(row):
    try:

        pdbA = graphid_to_pdb_chains_dict[row["uni1"]]
        pdbB = graphid_to_pdb_chains_dict[row["uni2"]]

        curr_int = (pdbA,pdbB)

        negative_links.append(curr_int)

        return pd.Series([pdbA,pdbB])
        
    except KeyError:
        # print(row.uni1, row.uni2, "not in positive network")
        return pd.Series([np.nan,np.nan])
    
print("Mapping graph ids from negative links dataset to specific PDB chain IDs")

negatome_df[['chain1','chain2']] = negatome_df.progress_apply(map_chains,axis=1)
neg_net_df = negatome_df.dropna()

print("Total unique negative interactions after chain mapping: ", neg_net_df.shape[0])

# %%
#filter out our initial unprot mapping dataframe to include info for those only having experimental evidence of interaction

unis_int = list(set(pos_net_df.uni1.values).union(set(pos_net_df.uni2.values)))
filtered_uniprot_df = uniprot_df[uniprot_df['Entry'].isin(unis_int)]

uni_pdb_ids = sorted(list(set(pos_net_df.chain1.values).union(set(pos_net_df.chain2.values))))

#download different protein representations of different protein files

# %run pdb_to_representations.py
download_protein_representations(uni_pdb_ids, seq_dict, pdb_chains_dname)

# %%
#create node feature set and filter current pdb id list to only those with node features
random.Random(88).shuffle(uni_pdb_ids)


filtered_pdb_ids = []
graphs = []
llms = []
anm_modes = []

for pdb_id in tqdm(uni_pdb_ids, desc='Creating protein features dataset'):
    try:
        
        graph = load_graphs(f'../graphs/{pdb_id}.pt')[0][0]
        anm_mode =torch.load(f"../anm_modes/{pdb_id}.pt")
        llm =torch.load(f"../llms/{pdb_id}.pt")

        graphs.append(graph)
        anm_modes.append(anm_mode)
        llms.append(llm)

        filtered_pdb_ids.append(pdb_id)

    except Exception as e:
        continue
        # print(e)
        # print(pdb_id)

torch.save(filtered_pdb_ids, f"../data/{prefix}_pdb_chains_list.pt")

print("PDB chains list saved to ",  f"../data/{prefix}_pdb_chains_list.pt")

#map pdb chain id to node id
pdb_chains_to_nodeid_dict = {}

for node_id, pdb_id in enumerate(filtered_pdb_ids):
    pdb_chains_to_nodeid_dict[pdb_id] = node_id
    

# %%
#splitting
filtered_pdb_ids = np.array(filtered_pdb_ids)

if eval_fold:

    train_val_nodes = filtered_pdb_ids

    kf = KFold(n_splits=5, shuffle=True, random_state=88)

    kf_split_indices = kf.split(train_val_nodes)
    
else:

    total_num_graphids = len(filtered_pdb_ids)

    mid_tresh = round(total_num_graphids*0.25)

    #70% for training and validation node set
    train_val_nodes = filtered_pdb_ids[mid_tresh:]
    test_nodes = filtered_pdb_ids[:mid_tresh]

    tresh = round(len(train_val_nodes)*0.2)

    kf = KFold(n_splits=4, shuffle=True, random_state=88)

    kf_split_indices = kf.split(train_val_nodes)

training_set = []
validation_set = []

for i, (train_index, val_index) in enumerate(kf_split_indices):
    train_nodes = train_val_nodes[train_index]
    val_nodes = train_val_nodes[val_index]

    pos_train_source_node = []
    pos_val_source_node = []

    neg_train_source_node = []
    neg_val_source_node = []

    #create positive training and validation links 
    for pairs in tqdm(positive_links, desc=f"Creating positive training and validation links for split {i+1}"):

        try:
            nodeA = pdb_chains_to_nodeid_dict[pairs[0]]
            nodeB = pdb_chains_to_nodeid_dict[pairs[1]]

            if pairs[0] in train_nodes and pairs[1] in train_nodes:
                pos_train_source_node.append((nodeA,nodeB))
                pos_train_source_node.append((nodeB,nodeA))
    #             train_source_node.append((nodeB,nodeA))

            elif pairs[0] in val_nodes and pairs[1] in val_nodes:
                pos_val_source_node.append((nodeA,nodeB))
                pos_val_source_node.append((nodeB,nodeA))
    #             val_source_node.append((nodeB,nodeA))

        except KeyError:
            continue

    #create negative training and validation links 
    for pairs in tqdm(negative_links, desc=f"Creating negative training and validation links for split {i+1}"):

        try:

            nodeA = pdb_chains_to_nodeid_dict[pairs[0]]
            nodeB = pdb_chains_to_nodeid_dict[pairs[1]]

            if pairs[0] in train_nodes and pairs[1] in train_nodes:
                neg_train_source_node.append((nodeA,nodeB))
                neg_train_source_node.append((nodeB,nodeA))
    #             train_source_node.append((nodeB,nodeA))

            elif pairs[0] in val_nodes and pairs[1] in val_nodes:
                neg_val_source_node.append((nodeA,nodeB))
                neg_val_source_node.append((nodeB,nodeA))
    #             val_source_node.append((nodeB,nodeA))

        except KeyError:
            continue

    # print("total positive links with voxels in training set: ", len(pos_train_source_node))
    # print("total positive links with voxels in validation set: ", len(pos_val_source_node))
    # print("total negative links with voxels in training set: ", len(neg_train_source_node))
    # print("total negative links with voxels in validation set: ", len(neg_val_source_node))

    #get the minimum number of edges we can cut down to
    min_num_train_edges = min(len(pos_train_source_node),len(neg_train_source_node))
    min_num_val_edges = min(len(pos_val_source_node),len(neg_val_source_node))

    positive_train_edges = pos_train_source_node[:min_num_train_edges]
    negative_train_edges = neg_train_source_node[:min_num_train_edges]
    positive_val_edges = pos_val_source_node[:min_num_val_edges]
    negative_val_edges = neg_val_source_node[:min_num_val_edges]

    print(f"minimum positive links with voxels in training set split {i+1}: ", len(positive_train_edges))
    print(f"minimum negative links with voxels in training set: {i+1}", len(negative_train_edges))
    print(f"minimum positive links with voxels in validation set: {i+1}", len(positive_val_edges))
    print(f"minimum negative links with voxels in validation set: {i+1}", len(negative_val_edges))

    train_edge = positive_train_edges + negative_train_edges
    val_edge = positive_val_edges + negative_val_edges

    # save the edges and node features for tuning

    training_set.append(torch.tensor(train_edge))
    validation_set.append(torch.tensor(val_edge))


torch.save(training_set,f"../data/{prefix}_train_interactions.pt")
torch.save(validation_set,f"../data/{prefix}_val_interactions.pt")

print("Training set saved to ",  f"../data/{prefix}_train_interactions.pt")
print("Validation set saved to ",  f"../data/{prefix}_val_interactions.pt")

save_graphs(f"../data/{prefix}_at_dgs_feat.pt", graphs)
torch.save(torch.stack(anm_modes), f"../data/{prefix}_anm_modes_feat.pt")
torch.save(torch.stack(llms), f"../data/{prefix}_llm_feat.pt")
print("Protein atomic graph features saved to ",  f"../data/{prefix}_at_dgs_feat.pt")
print("Protein ANM features  saved to ",   f"../data/{prefix}_anm_modes_feat.pt")


if not eval_fold:

    pos_test_source_node = []
    neg_test_source_node = []

    for pairs in tqdm(positive_links, desc="Creating positive testing links"):

        try:
            nodeA = pdb_chains_to_nodeid_dict[pairs[0]]
            nodeB = pdb_chains_to_nodeid_dict[pairs[1]]

            if pairs[0] in test_nodes and pairs[1] in test_nodes:
                pos_test_source_node.append((nodeA,nodeB))
                pos_test_source_node.append((nodeB,nodeA))

        except KeyError:

            continue

    # #create negative training and validation links 
    for pairs in tqdm(negative_links, desc="Creating negative testing links"):

        try:
            nodeA = pdb_chains_to_nodeid_dict[pairs[0]]
            nodeB = pdb_chains_to_nodeid_dict[pairs[1]]

            if pairs[0] in test_nodes and pairs[1] in test_nodes:
                neg_test_source_node.append((nodeA,nodeB))
                neg_test_source_node.append((nodeB,nodeA))

        except KeyError:

            continue


    # print("total positive links with voxels in testing set: ", len(pos_test_source_node))
    # print("total negative links with voxels in testing set: ", len(neg_test_source_node))

    #get the minimum number of edges we can cut down to
    min_num_test_edges = min(len(pos_test_source_node),len(neg_test_source_node))

    positive_test_edges = pos_test_source_node[:min_num_test_edges]
    negative_test_edges = neg_test_source_node[:min_num_test_edges]

    print("minimum positive links with voxels in testing set: ", len(positive_test_edges))
    print("minimum negative links with voxels in testing set: ", len(negative_test_edges))

    test_edge = positive_test_edges + negative_test_edges

    torch.save(torch.tensor(test_edge),f"../data/{prefix}_test_interactions.pt")
    print("Testing set saved to ",  f"../data/{prefix}_test_interactions.pt")
