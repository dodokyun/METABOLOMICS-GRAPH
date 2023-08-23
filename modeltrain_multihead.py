
import argparse
parser = argparse.ArgumentParser()
# graph generation parameter

parser.add_argument("-c", "--cutoff",type=float, dest="cutoff", action="store") 
parser.add_argument("-e", "--edgeweight", action='store_true')
parser.add_argument("-d", "--diseaseN", type=int, dest='diseaseN', action="store")

# model parameter 
parser.add_argument("-p", "--path", action = "store")
parser.add_argument("-l", "--loadmodel", action="store_true")
parser.add_argument("-m", "--savemodel", action="store_true")
parser.add_argument('-s', "--l")
args = parser.parse_args()

cutoff = args.cutoff
edge_weight = args.edgeweight # T/F
model_path = args.path
loadmodel = args.loadmodel
diseaseN = args.diseaseN
savemodel = args.savemodel


# multi head without t2d 

import pandas as pd 
import numpy as np 
import os
import json
import random 
import math
import torch
from MetabData import MetabDataset, MetabDataLoader
from GraphData import GraphDataset, GraphDataLoader
from Trainer import trainer_multidisease
from multiD_model import *
#from Trainer2 import *
from Lossfxn import CoxPHloss
import Models

gsds = '/home/leelabsg/media/leelabsg-storage0/UKBB_WORK/METABOLOMICS_WORK'
dokyun = '/Users/dokyunkim/Desktop/연구/UKBB'
newgsds = '/home/leelabsg/media/leelabsg-storage0/dokyun/UKBB/'
dk ='/home/n1/dokyunkim/UKBB_METAB_FINAL'
############# import data dictionary 
path = dk
gset = GraphDataset(root=dk +'/Data/',datapath = 'multi_disease_dict2')


opt1 = {'mtype'       : 'all',
       'winsorize'   :  3,
       'dtype'       :  'multi_disease', #'t2d-cad',
       'howtomerge'  : 'inner',
       'scaler'      :  'normalize',
       'cov'         : ['base', 'physical', 'health','clinical', 'socio'],
       'incidence'   : None,
       'tte'         : None,
       'exclude'     : ['EUR','tte_year'] 
      }



dat = gset.load_data(opt1, xy_split=False)



### single head #### 



print('multi_target: ',flush=True)
target = ['Dementia','MACE','Liver_Disease','Renal_Disease','Atrial_Fibrillation','Heart_Failure','CHD','Venous_Thrombosis',
    'Cerebral_Stroke','AAA','PAD','Asthma','COPD','Lung_Cancer','Skin_Cancer','Colon_Cancer','Rectal_Cancer','Parkinson',
   'Fractures','Cataracts','Glaucoma']
excols = ['t2d']



opt2 = {'data'           : dat,
        'output_col'     : 'multi_disease',
        'tte_col'        : 'multi_disease',
        'adj_matrix'     : 'gpcorr',
        'edge_weight'    : edge_weight,
        'cutoff'         : cutoff,
        'pre_transform'  : None,
        'cov_sep'        : False,
        'exclude_metab'  : None,
        'exclude_disease': excols
      }
graphs = gset.graphgenerator(opt2)
tasks = target

# Network Parameters 
GNNmodules = ['GCNConv','GraphConv','GraphConv']
feature_size = graphs[0].num_features
hidden_channel1 = 256
hidden_channel2 = 64
out_channels = 2
n_layers=3
task_names = tasks
dropout =0.2 # this need to be float 
norm_fn='nn.BatchNorm1d'
activation='nn.SELU'
aggr='max'

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model = MultiDiseaseNetwork3(GNNmodules, 
                feature_size,
                 hidden_channel1,
                 hidden_channel2,
                 out_channels,
                 n_layers,
                 task_names,
                 dropout, 
                 norm_fn, 
                 activation, 
                 aggr)

model_name =f'disease=multihead-cut_off={cutoff}-edge_weight={edge_weight}'
#model_save = model_path+'/'+model_name#+'.pt'


if loadmodel:
    saved_model_path = torch.load(model_path+'/parameter_entireModel.pt')
    model.load_state_dict(saved_model_path, strict=False)
    print('pretrained model loaded', flush =True)




print(model_name)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)
print(model)


optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.96 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)

# trainer parameters 
#model, 
graph_list=graphs

split_ratio =[0.8,0.1,0.1] 
num_classes=out_channels
ordered_tasks = tasks
seed = 12345
batch_size=128
epoch = 100
mask = 100
inference_batch=128

print(model_name)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)
print(model)

tt = trainer_multidisease( model=model, graph_list=graph_list,num_classes=num_classes,split_ratio=split_ratio, ordered_tasks= ordered_tasks, mask=mask,device = device, optimizer=optimizer,inference_batch=inference_batch,batch_size=batch_size, epoch =epoch, path=model_path, savemodel=savemodel)
tt.forward()
