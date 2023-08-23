import pandas as pd 
import numpy as np 

import torch

from MetabData import MetabDataset, MetabDataLoader
from GraphData import GraphDataset, GraphDataLoader
from trainer import trainer

### Path 
dokyun = '/Users/dokyunkim/Desktop/연구/UKBB'
path = dokyun 
os.chdir(path)


############ Generate Dataset 
# Import data dictionary which contains data sets 
data = MetabDataset(root = path + '/UKBB_METAB_UTILS/', datapath ='original_datadict2')
data.dset.keys()

>> ['met0','met3', 'met6','bc0', 'bc3' ,'bc6',  # metabolites and blood-chemistry data. the number means the winsorization threshold, outliers outisde 'n'-IQR are replaced with the max/min value
	'cov_w_drug', 'cov_wo_drug', 'norelated',  # covariates with/without drug consumption / norealted : European Ancestry who are genetically close are removed, norelatedness
	 't2d', 'cad', 't2d-cad-raw', 't2d-cad'  # t2d and cad data, prevalence incidence time-to-event recorded and t2d-cad merged data 
	 'regular_corr_bonferroni', 'regualr_pcorr_bonferroni', 'genetic_corr_bonferroni', 'genetic_pcorr_bonferroni'] # regualr/genetal correlation and partial correlation with statistically significant with bonferroni correction


# generate processed dataset with the given options
opt1 = {'mtype'       : 'all', 
       'winsorize'   :  3,
       'dtype'       : 'cad',
       'scaler'      :  'standardize',
       'cov'         :  ['base', 'physical', 'health','clinical', 'socio'],
       'incidence'   : 'cad',
       'tte'         : 'cad_year',
       'exclude'     : ['EUR']
      }

# if xy_split = False, it returns dataframe / if it is True, it returns input(features) / incid(label) / tte(time-to-event) chunks 
dat1 = data.load_data(opt,xy_split= False)
dat2 = data.load_data(opt,xy_split= True)

# add or remove data from data dictionary 

data.add_dataset(datadict, datapath=None) # input: datadict = {'name1':dataframe1, 'name2':dataframe2...} / datapath == None, it overwrites on the existing datapath 
data.remove_dataset(names, datapath=None) # inpit: names = ['name1','name2',...], type a list of data set name (can be chcked data.dset.keys()) / / datapath == None, it overwrites on the existing datapath 


### Call DataLoader using generated dataset 
opt2 = {'data'    : dat1,            # dat1 should go not dat2 , means xy_split =False 
		'ratio'  : [0.8,0.1,0.1]     # train val test ratio, sum to 1 
		'batch_size': 256,           # train_loader batch size 
		'inference_batch': 256, 	 # val and test loader batch size 
		'seed'   : 1234}    		 # random seed

loader = MetabDataLoader(opt)
train_loader, val_loader, test_loader = loader.data_loader()





############ Generate Graph Dataset 
# Import data dictionary which contains data sets  
data = GraphDataset(root=dk +'/Data/',datapath = 'graph_dict2')


# generate processed dataset with the given options
opt3 = {'mtype'       : 'all',
       'winsorize'   :  3,
       'dtype'       : 't2d-cad',
       'scaler'      :  None,
       'cov'         : ['base', 'physical', 'health','clinical', 'socio'],
       'incidence'   : 'status',
       'tte'         : 'tte_days',
       'exclude'     : ['EUR','tte_year']
      }

dat3 = data.load_data(opt3, xy_split=False) # this returns the dataframe as dat1 above 
 
# generate graph 
opt4 = {'data'           : dat1,
        'output_col'     : ['status'],
        'adj_matrix'     : 'gpcorr',
        'edge_weight'    : False,
        'cutoff'         : 0.8,
        'pre_transform'  : None
      }
graphs = data.graphgenerator(opt4) # this return a list of graphs 


opt5 = {'graph_list'     : graphs,
        'ratio'          : [0.8,0.1,0.1],
        'batch_size'     : 256,
        'inference_batch': 256,
        'seed'           : 100}


loader = GraphDataLoader(opt111)
train_loader, val_loader, test_loader = loader.data_loader()

