import torch
import pandas as pd
import math
import numpy as np
import torch
from urllib import request
from os.path import exists
import time
import os
import subprocess
import argparse
from six.moves.urllib.request import urlopen
import torch.nn.functional as F
import json
from tqdm import tqdm
from pdb_to_representations import download_protein_representations 
from dgl.data.utils import load_graphs, save_graphs

mapping = pd.read_csv("../data/multi_mapping.tsv", sep="\t")

pdbs = ['1J6Z',
 '1ATN',
 '1FIN',
 '1NBF',
 '1XD3',
 '2G45',
 '2HTH',
 '4UN2',
 '1G0X',
 '1P7Q',
 '1HCL',
 '1BI7',
 '2FIF']

pdb_chains_dname = '../h_parsed_pdbs'
hydrated_pdbs_dname = '../h_pdbs'
pdbs_dname = '../pdbs'

def add_hydrogens(pdb_id):
    finished = False
    # Define the command to run and the maximum amount of time to allow it to run for
    
    command = f'hydride -n 100 -i {pdbs_dname}/{pdb_id}.pdb -o {hydrated_pdbs_dname}/{pdb_id}.pdb'
    timeout = 300  # in seconds

    # Start the process and set a timer
    proc = subprocess.Popen(command, shell=True)
    timer = time.time()

    # Poll the process for as long as the timer has not expired
    while (time.time() - timer) < timeout:
        if proc.poll() is not None:
            if proc.returncode == 0:
                finished = True
            break
        time.sleep(1)

    # If the process is still running, terminate it
    if proc.poll() is None:
        proc.terminate()
        
    return finished
    
# download all unique pdb files, get protein content info and hydrate the protein

pdb_chains_dict = {}

for pdb_id in tqdm(pdbs, desc="Downloading and hydrating pdb file"):

    try:
        url = "https://files.rcsb.org/download/" + pdb_id + ".pdb"
        request.urlretrieve(url, f"{pdbs_dname}/{pdb_id}.pdb")
        
    except Exception as e:
        print(e)
        print(f'PDB file download issue with {pdb_id}')
        continue

    try:
        content = urlopen('https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/' + pdb_id).read()
        content = json.loads(content.decode('utf-8'))[pdb_id.lower()]['UniProt']
        pdb_chains_dict[pdb_id] = content
    except Exception as e:
        print(f'PDB chain info not available for {pdb_id}')
    
    if not add_hydrogens(pdb_id):
        print(f'Cannot hydrate {pdb_id}')

def find_chain(r):

    pdb_id, uni1 = r.From, r.Entry

    content = pdb_chains_dict[pdb_id]

    chains_1 = ""
    command_chains_1 = ""

    for c in sorted(list(set([entity['chain_id'] for entity in content[uni1]['mappings']]))):
        chains_1 += c
        command_chains_1 += c+','
    
    command_chains_1 = command_chains_1[:-1]
    pdb_chain_1 =  pdb_id + chains_1

    com = os.system(f"pdb_selchain -{command_chains_1} {hydrated_pdbs_dname}/{pdb_id}.pdb | pdb_tidy > {pdb_chains_dname}/{pdb_chain_1}.pdb")

    if not os.path.exists(f'{pdb_chains_dname}/{pdb_chain_1}.pdb'):
        # print('selchain does not work for ', pdb_id, "when extracting", command_chains_1)
        raise Exception('selchain does not work')

    return pdb_chain_1

mapping["PDB_Chain"] = mapping.apply(find_chain, axis=1)

links = pd.read_csv("../data/multi_test_links.txt")

total_links = []

chains = set()
chains.update(list(links.pdb1.unique()))
chains.update(list(links.pdb2.unique()))
pdb_chains = sorted(list(chains))

torch.save(pdb_chains, f"../data/multi_pdb_chains_list.pt")

def map_chains(row):

    pdbA = row.pdb1
    pdbB = row.pdb2

    curr_int = (pdbA,pdbB)

    total_links.append(curr_int)

links.apply(map_chains, axis=1)

multi_seq_dict = dict(zip(mapping['PDB_Chain'], mapping['Sequence']))

download_protein_representations(pdb_chains, multi_seq_dict, pdb_chains_dname)

#map pdb chain id to node id
pdb_chains_to_nodeid_dict = {}

for node_id, pdb_id in enumerate(pdb_chains):
    pdb_chains_to_nodeid_dict[pdb_id] = node_id

test_source_node = []

for pairs in tqdm(total_links, desc=f"Creating links"):

    nodeA = pdb_chains_to_nodeid_dict[pairs[0]]
    nodeB = pdb_chains_to_nodeid_dict[pairs[1]]

    test_source_node.append((nodeA,nodeB))
    test_source_node.append((nodeB,nodeA))

graphs = []
anms = []
llms = []

graph_folder ='../graphs'
anm_mode_folder = '../anm_modes'
llm_folder = '../llm'

for pdb_id in tqdm(pdb_chains, desc="Creating protein representations"):

    dg = load_graphs(f'{graph_folder}/{pdb_id}.pt')[0][0]
    anm = load_graphs(f'{anm_mode_folder}/{pdb_id}.pt')[0][0]
    llm = torch.load(f'{llm_folder}/{pdb_id}.pt')
        
    llms.append(llm)
    graphs.append(dg)
    anms.append(anm)

torch.save(torch.tensor(test_source_node),f"../data/multi_test_interactions.pt")
save_graphs(f"../data/multi_at_dgs_feat.pt", graphs)
save_graphs(f"../data/multi_dyn_graphs_feat.pt", anms)
torch.save(torch.stack(llms), f"../data/multi_llm_feat.pt")