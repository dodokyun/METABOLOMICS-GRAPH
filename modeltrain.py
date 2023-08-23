import argparse
parser = argparse.ArgumentParser()
# graph generation parameter
parser.add_argument("-d", "--disease", dest="disease", action="store")  
parser.add_argument("-c", "--cutoff",type=float, dest="cutoff", action="store") 
parser.add_argument("-e", "--edgeweight", action='store_true')
parser.add_argument("-s", "--covsep", action="store_true") 

# model parameter 
parser.add_argument("-m", "--model", dest="model", action="store") 
parser.add_argument("-ch", "--hchannels",type=int, dest="hchannels", action="store") 
parser.add_argument("-ne", "--epochs", type=int, action = "store")
parser.add_argument("-p", "--path", action = "store")
parser.add_argument("-l", "--loadmodel", action="store_true")
parser.add_argument("-dr", "--dropout", type=int, action="store")
args = parser.parse_args()


disease = args.disease 
cutoff = args.cutoff
cov_sep = args.covsep # T/F
edge_weight = args.edgeweight # T/F
disease = args.disease 
hidden_channels = args.hchannels
epoch = args.epochs 
model_path = args.path
model_load = args.loadmodel
dropout = args.dropout


import pandas as pd 
import numpy as np 
import os
import json
import random 
import math
import torch



from MetabData import MetabDataset, MetabDataLoader
from GraphData import GraphDataset, GraphDataLoader
from Trainer import trainer, trainer_cox
#from Trainer2 import *
from Lossfxn import CoxPHloss
import Models

gsds = '/home/leelabsg/media/leelabsg-storage0/UKBB_WORK/METABOLOMICS_WORK'
dokyun = '/Users/dokyunkim/Desktop/연구/UKBB'
newgsds = '/home/leelabsg/media/leelabsg-storage0/dokyun/UKBB/'
dk ='/home/n1/dokyunkim/UKBB_METAB_FINAL'

############# import data dictionary 


path = dk
gset = GraphDataset(root=dk +'/Data/',datapath = 'graph_dict2')


opt1 = {'mtype'       : 'all',
       'winsorize'   :  3,
       'dtype'       :  disease, #'t2d-cad',
       'scaler'      :  None,
       'cov'         : ['base', 'physical', 'health','clinical', 'socio'],
       'incidence'   : disease,
       'tte'         : 'tte_days',
       'exclude'     : ['EUR','tte_year']
      }

dat = gset.load_data(opt1, xy_split=False)
'''
if disease =='t2d':
	dat = pd.read_csv("./t2d_dat.csv")
if disease =='cad':
	dat = pd.read_csv("./cad_dat.csv")
'''

# edge = True / cov_sep = False

opt2 = {'data'           : dat,
        'output_col'     : disease,
        'tte_col'        : 'tte_days',
        'adj_matrix'     : 'gpcorr',
        'edge_weight'    : True,
        'cutoff'         : cutoff,
        'pre_transform'  : None,
        'cov_sep'        : cov_sep,
        'exclude_metab'  : None
      }

graphs = gset.graphgenerator(opt2)


torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


num_classes = 2

module = __import__('Models')
module = getattr(module, args.model)
model_ = module(feature_size=graphs[0].num_features, num_classes=num_classes, hidden_channels=hidden_channels,dropout= dropout)



optimizer = torch.optim.Adam(model_.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.96 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)
if disease == 't2d':
	weights = torch.tensor([0.5,12],dtype=torch.float).to(device)
if disease == 'cad': 
	weights = torch.tensor([0.5,7],dtype=torch.float).to(device)

criterion =torch.nn.CrossEntropyLoss(weight= weights)#,label_smoothing=0.1)

model_name =f'disease={disease}-model={args.model}-cut_off={cutoff}-edge_weight={edge_weight}-cov_sep={cov_sep}-hchannels={hidden_channels}'

model_save = model_path+'/'+model_name#+'.pt'

if model_load:
    model_.load_state_dict(torch.load(model_save))


print(model_name)
pytorch_total_params = sum(p.numel() for p in model_.parameters())
print(pytorch_total_params)
print(model_)

tt = trainer( model=model_, graph_list=graphs,num_classes=num_classes,split_ratio=[0.8,0.1,0.1], device = device, optimizer=optimizer,path=model_save, inference_batch=256, criterion=criterion,batch_size=256, epoch =epoch)
tt.forward()


