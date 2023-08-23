import numpy as np 
import pandas as pd
import torch
from torch import sparse
from torch.utils.data import Dataset, DataLoader,  random_split
import functools as ft
from sklearn import preprocessing
import math
import click
from MetabData import MetabDataset

from torch_geometric.loader import DataLoader as gDataLoader
from torch_geometric.data import Data


import os

# torch spase error 
#!pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_sparse-0.6.17+pt20cu118-cp311-cp311-linux_x86_64.whl


class GraphDataset(MetabDataset):
    def __init__(self,root, datapath):
        super().__init__()
        
        self.root        = root 
        self.datapath    = datapath
        self.dset        = torch.load(self.root + self.datapath)  

        self.adj_columns  = None
        self.graph_list   = None
        self.adjacency    = None

        # dlist is used only when input data is 'multi_disease'
        self.dlist    = ['Dementia','MACE','t2d','Liver_Disease','Renal_Disease','Atrial_Fibrillation','Heart_Failure','CHD','Venous_Thrombosis','Cerebral_Stroke',
                        'AAA','PAD','Asthma','COPD','Lung_Cancer','Skin_Cancer','Colon_Cancer','Rectal_Cancer','Parkinson','Fractures','Cataracts','Glaucoma']
        # t2d in dlist is from the original t2d data (filtered with multiple sources, not only with ICD10)
        # other diseases are defined from ICD10 only 

       
       # This will generate a list of diseases, which will be used for output_col. 
       # It indicates the order of diseases/tasks of output_col thus, we can use the order as names of tasks for the output of model layer, which is a dictionary
        self.disease_order = None 

    def graphgenerator(self, opt):
        
        # data and output column 
        data             = opt['data']                 # data frame
        edge_weight      = opt['edge_weight']          # include edge weight True/False
        pre_transform    = opt['pre_transform']        # pre_transform function 
        cov_sep          = opt['cov_sep']              # True: Separate, assign covariates attributes to graph data, not to node feature  / False: Not separate, include covaraites in node feature 
        adj_matrix       = opt['adj_matrix']           # type of ajacency matrix <'corr','pcorr','gcorr','gpcorr'>
        
        if opt['output_col'] == opt['tte_col'] == 'multi_disease':
            multi_disease_flag=True
            print('Multi_disease_flag is True')
            
            exclude_disease  = opt['exclude_disease']
            if isinstance(exclude_disease, list)==False:
                exclude_disease = [exclude_disease]
            
            # exclude diseases from dlist (which will be used as output/tte columns)
            self.dlist2 = [d for d in self.dlist if d not in exclude_disease]
            print('diseases in exclude_disease are removed from dlist')
        
        else:
            if opt['output_col'] != 'multi_disease' and opt['tte_col'] != 'multi_disease':
                multi_disease_flag=False
            else:
                raise ValueError('Check "output_col" and "tte_col"')

        # column name of time-to-event ex) tte_year/tte_days
        output_col       = self.dlist2 if multi_disease_flag else opt['output_col']                      
        # column name of label ex) t2d/cad/status 
        tte_col          = [d+'_tte' for d in self.dlist2] if multi_disease_flag else opt['tte_col']       

        if isinstance(output_col, list)==False:
            output_col = [output_col]
        if isinstance(tte_col, list)==False:
            tte_col = [tte_col]
        
        # adjacency matrix 
        if adj_matrix in ['corr','pcorr','gcorr','gpcorr','gip']:
            self.adjacency = self.adjacency_matrix(opt)
        else: 
            raise ValueError("choose one from ['corr','pcorr','gcorr','gpcorr','gip']")
        
        # edge index from adjacency matrix
        edge_index = torch.tensor(self.adjacency.iloc[:,:2].values).t().contiguous()
        if edge_weight: 
            edge_weight = torch.tensor(self.adjacency.iloc[:,2].values).reshape(1,len(self.adjacency)).t().to(torch.float32)
        else:
            edge_weight = None
        
        # inputs = metabolite columns only 
        inputs = torch.tensor(data[self.adj_columns].values).to(torch.float32)
        
        if multi_disease_flag:
            # outputs = can be multiple columns given by a list 
            outputs = {c: torch.tensor(data[c].values).to(torch.long) for c in output_col}
            # time-to-event column
            time_to_event = {c: torch.tensor(data[c].values).to(torch.float16) for c in tte_col}

        else: 
            # outputs = can be multiple columns given by a list 
            outputs = torch.tensor(data[output_col].values).to(torch.long)
            # time-to-event column
            time_to_event = torch.tensor(data[tte_col].values).to(torch.float16)
        
        # covaraites will be used as node features or as extra information 
        not_cov_cols = ['f.eid','cad','t2d','status','tte_days','tte_year']+self.adj_columns+output_col+tte_col+self.dlist+[d+'_tte' for d in self.dlist] 
        covariates = data.loc[:,~data.columns.isin(not_cov_cols) ] 
        covariates = torch.tensor(covariates.values).to(torch.float32)
    

        graph_list = []

        if cov_sep:
            for i in range(len(inputs)):
                # node feature
                x = torch.eye(len(inputs[i]))*inputs[i]
                covv = covariates[i].reshape([1,-1])
                
                if multi_disease_flag:
                    # true label
                    y =  {o: outputs[o][i] for o in outputs.keys()}
                    # time-to-event
                    tte = {t: time_to_event[t][i] for t in time_to_event.keys()}
                    # dataset
                    graph = Data(x=x, edge_index = edge_index, edge_weight = edge_weight, cov = covv, tte= tte, **y, **tte)

                else:
                    # true label
                    y = outputs[i]
                    # time-to-event
                    tte = time_to_event[i]
                    # dataset 
                    graph = Data(x=x, y=y, edge_index = edge_index, edge_weight = edge_weight, cov = covv, tte= tte)
            

                if pre_transform is not None: 
                    graph = pre_transform(graph)

                graph_list += [graph]
                self.graph_list = graph_list

        else: 
            for i in range(len(inputs)):
                # node feature
                x = torch.eye(len(inputs[i]))*inputs[i]
                x_feat = covariates[i].repeat(len(inputs[i]),1)
                x = torch.cat((x,x_feat),axis=1)
                
                if multi_disease_flag:
                    # true label
                    y =  {o: outputs[o][i] for o in outputs.keys()}
                    # time-to-event
                    tte = {t: time_to_event[t][i] for t in time_to_event.keys()}

                else:
                    # true label
                    y = outputs[i]
                    # time-to-event
                    tte = time_to_event[i]
                
                # dataset 
                graph = Data(x=x, y=y, edge_index = edge_index, edge_weight = edge_weight, tte = tte)

                if pre_transform is not None: 
                    graph = pre_transform(graph)

                graph_list += [graph]
                self.graph_list = graph_list
                
        return self.graph_list
    
    
    def adjacency_matrix(self, opt):
        atype  = opt['adj_matrix']
        cutoff = opt['cutoff']
        # select adjacency matrix type based on correlation or genetic correlation
        # correlation -> p-value correction with bonferroni correction (-> partial correction)
        exclude_metab = opt['exclude_metab']   
         
        if atype == 'corr':
            adj = self.dset['regular_corr_bonferroni']
        elif atype == 'pcorr':
            adj = self.dset['regualr_pcorr_bonferroni']
        elif atype =='gcorr':
            adj = self.dset['genetic_corr_bonferroni']
        elif atype =='gpcorr':
            adj = self.dset['genetic_pcorr_bonferroni']
        elif atype =='gip':
            adj = self.dset['gip']
        adj = adj.round(2)
        
        if exclude_metab is not None:
            if isinstance(exclude_metab, list)==False:
                exclude_metab = [exclude_metab]
            adj = adj.loc[~adj.columns.isin(exclude_metab),~adj.columns.isin(exclude_metab)]

        self.adj_columns = list(adj.columns)
        
        # dictionary { metab_id : numbering }
        d = dict(zip(adj.columns, [*range(len(adj.columns))]))
        
        # adjacency matrix -> list
        adjl = adj.rename_axis('source')\
                .reset_index()\
                .melt('source', value_name='weight', var_name='target')\
                .reset_index(drop=True)

        adjl = adjl[abs(adjl['weight']) > cutoff]
        adjl.loc[adjl['weight'] >1, 'weight'] = 1
        adjl.loc[adjl['weight'] <-1, 'weight'] = -1
        adjl = adjl.replace({'source':d, 'target':d})
        
        return adjl  

    def GIP_similarity(x,y, sigma):

        diff = x-y
        squared_distance = np.dot(diff, diff)
        return np.exp(-squared_distance / (2*sigma**2))           
            
    
class GraphDataLoader(Dataset):
    def __init__(self, opt,**kwargs):
        # ratio = [train, test, val], sum = 1
        self.ratio           = opt['ratio']
        self.seed            = opt['seed']
        self.graph_list      = opt['graph_list']
        self.len             = len(self.graph_list)
        self.batch_size      = opt['batch_size']
        self.inference_batch = opt['inference_batch']
        self.num_workers     = None
        
        if 'num_workers' in kwargs.keys():
            self.num_workers = kwargs['num_workers']

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
          
    def data_loader(self):
        crit = sum(self.ratio)
        if (crit > 1) or (crit < 1-1.0e-5):
            raise ValueError('train, test, validation ratio must sum to 1')

        if (self.ratio[0] == 0) or (self.ratio[1] == 0):
            raise ValueError('train and val cannot be zero')
    
        train_size = int(self.ratio[0] * self.len)
        val_size   = int(self.ratio[1] * self.len)
        test_size  = self.len - train_size - val_size
        
       
        # data split index 
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        
        train, test, val = random_split(self.graph_list, [train_size, test_size, val_size], generator = gen)
        
        val_batch =  len(val)  if self.inference_batch == None else self.inference_batch
        test_batch = len(test) if self.inference_batch == None else self.inference_batch
        
        if self.num_workers is not None:
            self.train_loader = gDataLoader(dataset = train, batch_size = self.batch_size , shuffle = True,num_workers=self.num_workers)
            self.val_loader   = gDataLoader(dataset = val,   batch_size = val_batch,   shuffle = False,num_workers=self.num_workers)
            self.test_loader  = gDataLoader(dataset = test,  batch_size = test_batch,  shuffle = False,num_workers=self.num_workers)
        else:
            self.train_loader = gDataLoader(dataset = train, batch_size = self.batch_size , shuffle = True)
            self.val_loader   = gDataLoader(dataset = val,   batch_size = val_batch,   shuffle = False)
            self.test_loader  = gDataLoader(dataset = test,  batch_size = test_batch,  shuffle = False)
        return self.train_loader, self.val_loader, self.test_loader
            









