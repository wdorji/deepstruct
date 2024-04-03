import pandas as pd
import numpy as np
import torch
from urllib import request
from os.path import exists
import time
import os
import subprocess
import argparse
from six.moves.urllib.request import urlopen
import json
from tqdm import tqdm

parser = argparse.ArgumentParser(description='download missing pdb files and hydrate them')
parser.add_argument('-ID_mapping', required=True, type=str, help='path to uniprot mapping file')
parser.add_argument('-prefix', required=True, type=str, help='prefix to save the ouput dictionary')
parser.add_argument('-hydrated_pdbs_dname', default='../../vox2net/h_pdbs', type=str, help='path to hydrated pdbs directory')
parser.add_argument('-pdbs_dname', default='../../vox2net/pdbs', type=str, help='path to pdbs directory')
parser.add_argument('-pos_int_data', default='../data/string_phy_human.txt', type=str, help='path to postive interaction dataset')
parser.add_argument('-STRING', action='store_true', help='A boolean argument to specify if the PPI is from STRING database, if so only interactions with greater than 70% confidence are used')

static_args = parser.parse_args()

intID_uni_seq_pdbs_fname = static_args.ID_mapping
out_path = f"../data/{static_args.prefix}_pdb_chains_dict.pt"
hydrated_pdbs_dname = static_args.hydrated_pdbs_dname
pdbs_dname = static_args.pdbs_dname
positive_network_fname = static_args.pos_int_data#"string_phy_homo_sapien.txt" ##ARG
STRING_DATABASE = static_args.STRING

pos_net_df = pd.read_csv(positive_network_fname,sep=" ").dropna()

#remove the reverse order duplicate
pos_net_df = pos_net_df.loc[pd.DataFrame(np.sort(pos_net_df[['protein1','protein2']],1),index=pos_net_df.index).drop_duplicates(keep='first').index]

if STRING_DATABASE:
    pos_net_df = pos_net_df[pos_net_df["combined_score"]>=700]

string_pbds_dict = {}
pdbs = set()

uniprot_df = pd.read_csv(intID_uni_seq_pdbs_fname, sep="\t").dropna()
def create_mapping(row):
        cur_pdbs = row["PDB"][:-1].split(";")

        string_pbds_dict[row["From"]] = cur_pdbs
        return row

uniprot_df.apply(create_mapping,axis=1)

def common_pdbs(r):
    try:
        pdbs1 = string_pbds_dict[r.protein1]
        pdbs2 = string_pbds_dict[r.protein2]

        pdbs.update(list(set(pdbs1).intersection(set(pdbs2))))
        return r
    except KeyError:
        return r

pos_net_df.apply(common_pdbs, axis=1)

#helper function to add missing hydrogen atoms to pdb file

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
    if not exists(f"{pdbs_dname}/{pdb_id}.pdb"):
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
        if len(content) < 2:
            continue
        pdb_chains_dict[pdb_id] = content
    except Exception as e:
        print(f'PDB chain info not available for {pdb_id}')

    if not os.path.exists(f'{hydrated_pdbs_dname}/{pdb_id}.pdb'):      
        if not add_hydrogens(pdb_id):
            print(f'Cannot hydrate {pdb_id}')
            
torch.save(pdb_chains_dict, out_path)

print("PDB to chains content dictionary saved at: ", out_path)