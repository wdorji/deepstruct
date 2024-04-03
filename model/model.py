import argparse
import os.path as osp
import seaborn as sns
import torch.nn.functional as F
import torch
import networkx as nx
import random
# import torch_geometric.transforms as T
import torch.nn as nn
from math import ceil
from torch import Tensor
from torchinfo import summary
from torch.utils.data import TensorDataset, DataLoader
# from torch_geometric.data import Data
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
# from voxel_conv_ae import ConvAutoencoder
# from torch_geometric.utils import negative_sampling, convert, to_dense_adj,  to_networkx
# from torch_geometric.nn import GCNConv, global_mean_pool, SAGEConv, global_add_pool
# from torch_geometric.nn.conv import MessagePassing
# from pyuul import VolumeMaker # the main PyUUL module
# from pyuul import utils # the PyUUL utility module
import time,os,urllib # some standard python modules we are going to use
import gc
import matplotlib.pyplot as plt
import numpy as np
import glob
# import torch_geometric.transforms as T
# from torch_geometric.datasets import Planetoid
# from torch_geometric.nn import GAE,GCNConv
# import torch_geometric.nn
# import optuna
import glob
import networkx as nx
import dgl
from dgl.data.utils import load_graphs, save_graphs
from dgllife.model.gnn.gcn import GCN
from dgllife.model.gnn.nf import NFGNN
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
from dgllife.model.readout.mlp_readout import MLPNodeReadout
from dgllife.model.readout.sum_and_max import SumAndMax
from dgllife.model.gnn import AttentiveFPGNN, GAT
from dgllife.model.readout import AttentiveFPReadout
import pandas as pd

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

set_seed(0)

cuda0 = torch.device('cuda:0')
device = torch.device("cpu")

class NeuralLinkPredictor(torch.nn.Module):
    
    ##### new architecture

    def __init__(self, graph_feat_size):
        super(NeuralLinkPredictor, self).__init__()

        #####encoder

        self.gnn = AttentiveFPGNN(node_feat_size=39,
                                  edge_feat_size=11,
                                  num_layers=3,
                                  graph_feat_size=graph_feat_size)
        
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=2)
        
        self.transform = nn.Linear(graph_feat_size, graph_feat_size)
        
        #####predictor
        pred_dim = [graph_feat_size*2+2048, 64, 1]

        self.pred = nn.ModuleList([nn.Linear(pred_dim[i], pred_dim[i+1]) for i in range(len(pred_dim)-1)])

        self.dropout = nn.Dropout(0.0)

    def reset_parameters(self):

        self.gnn.reset_parameters()
        self.transform.reset_parameters()

        for lin in self.pred:
            lin.reset_parameters()
    
    
    def encode_protein(self,x):##attentive FP
        
        x = x.to('cuda:0')

        node_feats = x.ndata['h']#.to('cuda:0')
        edge_feats = x.edata['e']#.to('cuda:0')

        node_feats = self.gnn(x, node_feats, edge_feats)

        node_feats = self.readout(x, node_feats, False)

        x = self.transform(node_feats)

        return x

    def forward(self, x1, x2, ax1, ax2):
        
        x1 = self.encode_protein(x1)
        x2 = self.encode_protein(x2)

        ax1 = ax1.to('cuda:0')
        ax2 = ax2.to('cuda:0')

        x = torch.cat([x1,ax1, x2, ax2],dim=1)

        for lin in self.pred[:-1]:
            x = F.relu(self.dropout(lin(x)))
    
        x = self.pred[-1](x)

        output = torch.sigmoid(x).squeeze()
        
        return output

    #######

# embd_name = "h_gnm"

average_metrics = []
average_multi_metrics = []

multi_edges = torch.load("../data/multi_test_interactions.pt")
multi_graph = np.array(load_graphs("../data/multi_at_dgs_feat.pt")[0])#.to(cuda0)
multi_anm_modes = torch.concat([torch.load("../data/multi_llm_feat.pt").to(device)], dim=1).float()#[:,:10]

multi_split = int(len(multi_edges)/2)

multi_pos_edge = multi_edges[:multi_split,:]
multi_neg_edge = multi_edges[multi_split:,:]
multi_label = torch.cat([torch.ones(multi_split, ), torch.zeros(multi_split, )], dim=0).long()#.to(cuda0)

test_edges = torch.load(f"../data/string_phy_human_test_interactions.pt")
test_split = int(len(test_edges)/2)

test_pos_edge = test_edges[:test_split,:]
test_neg_edge = test_edges[test_split:,:]
test_label = torch.cat([torch.ones(test_split, ), torch.zeros(test_split, )], dim=0).long()#.to(cuda0)

validation_set = torch.load(f"../data/string_phy_human_val_interactions.pt")

node_features = np.array(load_graphs("../data/string_phy_human_at_dgs_feat.pt")[0])#.to(cuda0)
anm_modes = torch.concat([torch.load("../data/string_phy_human_llm_feat.pt").to(device)], dim=1).float()#[:,:10]

training_set = torch.load(f"../data/string_phy_human_train_interactions.pt")

for i, (train_edges, val_edges) in enumerate(zip(training_set, validation_set)):

    val_split = int(len(val_edges)/2)

    val_pos_edge = val_edges[:val_split,:]
    val_neg_edge = val_edges[val_split:,:]
    val_label = torch.cat([torch.ones(val_split, ), torch.zeros(val_split, )], dim=0).long()#.to(cuda0)

    train_split = int(len(train_edges)/2)

    train_pos_edge = train_edges[:train_split,:]
    train_neg_edge = train_edges[train_split:,:]
    train_label = torch.cat([torch.ones(train_split, ), torch.zeros(train_split, )], dim=0)#.to(cuda0)
    

    def create_batch(pos_edges, neg_edges, perm):

        train_edge = torch.cat([pos_edges[perm,:], neg_edges[perm,:]], dim=0)

        # Our labels are all 1 for the positive edges and 0 for the negative ones                          
        pos_label = torch.ones(len(perm), )
        neg_label = torch.zeros(len(perm), )
        train_label = torch.cat([pos_label, neg_label], dim=0).to(cuda0)

        return dgl.batch(node_features[train_edge[:,0]]), dgl.batch(node_features[train_edge[:,1]]), anm_modes[train_edge[:,0]], anm_modes[train_edge[:,1]], train_label

    def metrics(pred, label, print_prob):
        label = label.cpu()
        pred = pred.cpu()
        pred_rounded = torch.where(pred>=0.5,1.,0.)
        if print_prob:
            print(label)
            print(pred)
            print(pred_rounded)
        accu = round(accuracy_score(label, pred_rounded)*100, 4)
        auc = round(roc_auc_score(label, pred_rounded)*100,4)
        ap = round(average_precision_score(label, pred_rounded)*100,4)
        f1 = round(f1_score(label, pred_rounded)*100, 4)
        pre = round(precision_score(label, pred_rounded)*100, 4)
        rec = round(recall_score(label, pred_rounded)*100, 4)
        mcc = round(matthews_corrcoef(label, pred_rounded), 4)
        print(confusion_matrix(label,pred_rounded))
        
        return [accu, auc, ap, f1, pre, rec, mcc]
        
    def val_loss(predictor, pos_edges, neg_edges, batch_size):
        with torch.no_grad():
            predictor.eval()

            epoch_val_loss = 0
            for perm in DataLoader(range(len(pos_edges)), batch_size=batch_size, shuffle=True):

                p1, p2, ap1, ap2, batch_label = create_batch(pos_edges, neg_edges, perm)
                preds = predictor(p1, p2, ap1, ap2)

                pred_loss_fn = torch.nn.BCELoss()

                loss = pred_loss_fn(preds, batch_label)

                epoch_val_loss += loss.item()

            epoch_val_loss = epoch_val_loss/ len(pos_edges)
        return epoch_val_loss

    def test_pred(predictor, pos_edges, neg_edges, batch_size, print_prob):
        with torch.no_grad():
            predictor.eval()

            total_preds = []
            total_labels = []
            for perm in DataLoader(range(len(pos_edges)), batch_size=batch_size, shuffle=True):

                p1, p2, ap1, ap2, batch_label = create_batch(pos_edges, neg_edges, perm)
                
                total_preds += [predictor(p1, p2, ap1, ap2)]#.reshape(1,)
                total_labels += [batch_label]

            total_preds = torch.cat(total_preds, dim = 0)
            total_labels = torch.cat(total_labels, dim = 0)

            # accu, auc, ap, f1 = metrics(total_preds, total_labels, print_prob)       
        return metrics(total_preds, total_labels, print_prob)    

    def test_multi_pred(predictor, print_prob):
        with torch.no_grad():
            predictor.eval()
            p1 = dgl.batch(multi_graph[multi_edges[:,0]])
            p2 = dgl.batch(multi_graph[multi_edges[:,1]])
            ap1 = multi_anm_modes[multi_edges[:,0]]
            ap2 = multi_anm_modes[multi_edges[:,1]]
            total_preds = predictor(p1, p2, ap1, ap2)

            # accu, auc, ap, f1 = metrics(total_preds, multi_label, print_prob)       
        return metrics(total_preds, multi_label, print_prob)

    losses = []
    val_losses = []
    test_acc = []
    multi_acc = []

    def single_run():
        print("SET: ", i+1)
        
        in_channels = 64

        lrate= 1e-6
        epochs = 100

        batch_size = 16

        predictor = NeuralLinkPredictor(in_channels).to(cuda0)

        pred_optimizer = torch.optim.Adam(predictor.parameters(), lr=lrate, weight_decay = 0)
        
        predictor.train()
        predictor.reset_parameters()
                
        for epoch in range(1,epochs + 1):

            epoch_total_loss = 0

            for perm in DataLoader(range(len(train_pos_edge)), batch_size=batch_size, shuffle=True):

                p1, p2, ap1, ap2, batch_label = create_batch(train_pos_edge, train_neg_edge, perm)

                preds = predictor(p1, p2, ap1, ap2)##.reshape(1,)

                pred_loss_fn = torch.nn.BCELoss()

                loss = pred_loss_fn(preds, batch_label)

                epoch_total_loss += loss.item()

                # Update our parameters

                pred_optimizer.zero_grad()

                loss.backward()

                pred_optimizer.step()
            
            epoch_total_loss = epoch_total_loss/ len(train_pos_edge)

            epoch_val_loss = val_loss(predictor, val_pos_edge, val_neg_edge, batch_size)

            test_metric = test_pred(predictor, test_pos_edge, test_neg_edge, batch_size, False)
            print(test_metric)

            multi_metric = test_multi_pred(predictor, True)
            print(multi_metric)

            losses.append(epoch_total_loss)
            val_losses.append(epoch_val_loss)
            test_acc.append(test_metric[0])
            multi_acc.append(multi_metric[0])

            print(f"Epoch: {epoch}, Training Loss: {epoch_total_loss}, Validation Loss: {epoch_val_loss}, Test Accuracy: {test_metric[0]}, Multi Accuracy: {multi_metric[0]}")


        average_metrics.append(test_metric)
        average_multi_metrics.append(multi_metric)

        epochs = range(1, len(losses) + 1)

        torch.save(predictor.state_dict(), f"atom_plm_64_{i+1}_model.pt")

        # Create a line plot for training and validation loss
        plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
        sns.lineplot(x=epochs, y=losses, label="Training loss")
        sns.lineplot(x=epochs, y=val_losses, label="Validation loss")

        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # plt.savefig(f"atom_plm_{i+1}_loss_curves")
        plt.show()

    single_run()

df = pd.DataFrame(average_metrics, columns = ["Accuracy", "AUC","AP","F1", "Precision", "Recall", "MCC"])
print(df)
print(df.mean())

d = pd.DataFrame(average_multi_metrics, columns = ["Multi Accuracy", "Multi AUC","Multi AP","Multi F1", "Multi Precision", "Multi Recall", "Multi MCC"])
print(d)
print(d.mean())

def round_4(a_list):
    return [round(num, 4) for num in a_list]

print(df.mean())
print(df.std())
print(round_4(df.mean().to_list()))
print(round_4(df.std().to_list()))
print()
print(d.mean())
print(d.std())
print(round_4(d.mean().to_list()))
print(round_4(d.std().to_list()))