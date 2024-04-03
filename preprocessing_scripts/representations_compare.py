import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
import matplotlib.pyplot as plt
import numpy as np
import random
import os

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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SimpleNN(torch.nn.Module):
    
    ##### new architecture

    def __init__(self, in_channels):
        super(SimpleNN, self).__init__()

        #####predictor
        
        dims = [in_channels, 256]

        self.prot = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])

        pred_dim = [dims[-1]*2, 256, 1]

        self.pred = nn.ModuleList([nn.Linear(pred_dim[i], pred_dim[i+1]) for i in range(len(pred_dim)-1)])

        self.dropout = nn.Dropout(0.0)

    def reset_parameters(self):
        
        for lin in self.prot:
            lin.reset_parameters()

        for lin in self.pred:
            lin.reset_parameters()
    
    def encode_protein(self,x):

        for lin in self.prot:
            x = F.relu(lin(x))

        return x

    def forward(self, x1, x2):
        
        x1 = self.encode_protein(x1)

        x2 = self.encode_protein(x2)

        x = torch.cat([x1, x2],dim=1)
        # x = x.to('cuda:0')

        for lin in self.pred[:-1]:
            x = F.relu(self.dropout(lin(x)))
    
        x = self.pred[-1](x)

        output = torch.sigmoid(x).squeeze()
        
        
        return output

    #######
    

input_dims = {"llm": 1024,"ctf":343, "voxel":32768}

multi_edges = torch.load("../../vox2net/multi_test_interactions.pt")

multi_split = int(len(multi_edges)/2)

multi_pos_edge = multi_edges[:multi_split,:]
multi_neg_edge = multi_edges[multi_split:,:]
multi_label = torch.cat([torch.ones(multi_split, ), torch.zeros(multi_split, )], dim=0).long()#.to(cuda0)

# test_edges = torch.load(f"../data/string_phy_human_test_interactions.pt")
test_edges = torch.load(f"../../vox2net/test_edges.pt")
test_split = int(len(test_edges)/2)

test_pos_edge = test_edges[:test_split,:]
test_neg_edge = test_edges[test_split:,:]
test_label = torch.cat([torch.ones(test_split, ), torch.zeros(test_split, )], dim=0).long()#.to(cuda0)

train_edges = torch.load(f"../../vox2net/train_edges_0.pt")

train_split = int(len(train_edges)/2)

train_pos_edge = train_edges[:train_split,:]
train_neg_edge = train_edges[train_split:,:]
train_label = torch.cat([torch.ones(train_split, ), torch.zeros(train_split, )], dim=0)#.to(cuda0)

def NormalizeData(data):
  min_vals = torch.min(data, dim=1, keepdim=True).values
  max_vals = torch.max(data, dim=1, keepdim=True).values
  return (data - min_vals) / (max_vals - min_vals)

use_random = False

for embd_name, in_channels in input_dims.items():
  print("Testing for proteins in the form of: ", embd_name)

  node_feat = NormalizeData(torch.load(f"../../vox2net/string_phy_human_{embd_name}_feat.pt").float().to(device))

  multi_node_features = NormalizeData(torch.load(f"../../vox2net/multi_{embd_name}_feat.pt").float().to(device))

  if use_random:
    multi_node_features = torch.rand_like(multi_node_features).float().to(device)
    node_feat = torch.rand_like(node_feat).float().to(device)

  def create_batch(pos_edges, neg_edges, perm):

      train_edge = torch.cat([pos_edges[perm,:], neg_edges[perm,:]], dim=0)

      # Our labels are all 1 for the positive edges and 0 for the negative ones                          
      pos_label = torch.ones(len(perm), )
      neg_label = torch.zeros(len(perm), )
      train_label = torch.cat([pos_label, neg_label], dim=0).to(device)

      return node_feat[train_edge[:,0]], node_feat[train_edge[:,1]], train_label

  def metrics(pred, label):
      label = label.cpu()
      pred = pred.cpu()
      pred_rounded = torch.where(pred>=0.5,1.,0.)
      accu = round(accuracy_score(label, pred_rounded)*100, 4)
      auc = round(roc_auc_score(label, pred_rounded)*100,4)
      ap = round(average_precision_score(label, pred_rounded)*100,4)
      f1 = round(f1_score(label, pred_rounded)*100, 4)
      pre = round(precision_score(label, pred_rounded)*100, 4)
      rec = round(recall_score(label, pred_rounded)*100, 4)
      mcc = round(matthews_corrcoef(label, pred_rounded), 4)
      print(confusion_matrix(label,pred_rounded))
      
      return [accu, auc, ap, f1, pre, rec, mcc]

  def test_pred(predictor, pos_edges, neg_edges, batch_size):
      with torch.no_grad():
          predictor.eval()

          total_preds = []
          total_labels = []
          for perm in DataLoader(range(len(pos_edges)), batch_size=batch_size, shuffle=True):

              ap1, ap2, batch_label = create_batch(pos_edges, neg_edges, perm)
              
              total_preds += [predictor(ap1, ap2)]#.reshape(1,)
              total_labels += [batch_label]

          total_preds = torch.cat(total_preds, dim = 0)
          total_labels = torch.cat(total_labels, dim = 0)
  
      return metrics(total_preds, total_labels)    

  def test_multi_pred(predictor):
      with torch.no_grad():
          predictor.eval()
          ap1 = multi_node_features[multi_edges[:,0]]
          ap2 = multi_node_features[multi_edges[:,1]]
          total_preds = predictor(ap1, ap2)


      return metrics(total_preds, multi_label)

  def single_run():

      lrate= 1e-5

      epochs = 100

      batch_size = 50

      predictor = SimpleNN(in_channels).to(device)

      pred_optimizer = torch.optim.Adam(predictor.parameters(), lr=lrate)
      
      predictor.train()
      predictor.reset_parameters()
              
      for epoch in range(1,epochs + 1):

          epoch_total_loss = 0

          for perm in DataLoader(range(len(train_pos_edge)), batch_size=batch_size, shuffle=True):

              ap1, ap2, batch_label = create_batch(train_pos_edge, train_neg_edge, perm)

              preds = predictor(ap1, ap2)#.reshape(1,)

              pred_loss_fn = torch.nn.BCELoss()

              loss = pred_loss_fn(preds, batch_label.float())

              epoch_total_loss += loss.item()

              # Update our parameters

              pred_optimizer.zero_grad()

              loss.backward()

              pred_optimizer.step()

          test_metric = test_pred(predictor, test_pos_edge, test_neg_edge, batch_size)

          multi_metric = test_multi_pred(predictor)

          print(f'Epoch: {epoch}, Testing Accuracy: {test_metric[0]}, Multi Accuracy: {multi_metric[0]}')

  single_run()