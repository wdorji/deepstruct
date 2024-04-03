import pandas as pd
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description='collect all PPI network ids for mapping')
parser.add_argument('-pos_int_data', default='../data/string_phy_human.txt', type=str, help='path to postive interaction dataset')
parser.add_argument('-STRING', action='store_true', help='A boolean argument to specify if the PPI is from STRING database, if so only interactions with greater than 70% confidence is used')
parser.add_argument('-prefix', required=True, type=str, help='prefix to save ouput txt file of PPI network IDS in data folder')

static_args = parser.parse_args()

positive_network_fname = static_args.pos_int_data
STRING_DATABASE = static_args.STRING
out_path = f"../data/{static_args.prefix}_PPI_IDS.txt"

#collect protein ids from the PPI network file

pos_net_df = pd.read_csv(positive_network_fname,sep=" ").dropna()

if STRING_DATABASE:
    pos_net_df = pos_net_df[pos_net_df["combined_score"]>=700]

PPI_network_ids = set(list(pos_net_df.protein1.unique()) + list(pos_net_df.protein2.unique()))

with open(out_path, "w") as f:
    for network_id in tqdm(PPI_network_ids, desc="Getting PPI graph IDs"):
        f.write(f"{network_id}\n")

print("PPI graph IDs file saved at: ", out_path)

