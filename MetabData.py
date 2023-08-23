import numpy as np 
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader,  random_split
import functools as ft
from sklearn import preprocessing
from sklearn.utils.class_weight import compute_class_weight
import math
import click

class MetabDataset(Dataset):
    def __init__(self, root = '/media/leelabsg-storage0/dokyun/UKBB_METAB_UTILS/', datapath ='original_datadict2' ):
        super(MetabDataset).__init__()

        '''
        This class is to load metabolomics datasets.
        how to use: 
            1. Firstly load datasets from the given path, root : directory of the path / datapath : dataset (torch.save, pickled)
            - dset = MetabDataset(root, datapath)
        
            Now datasets are loaded
            you can access to datasets by calling 'dset.dset'
            To fully utilize the pre-proceseed datasets, need to call 'load_data' method 
            
            2. call 'load_data' function with options
            - dset.load_data(opt, xy_split)
            
            -> there are several options to choose and to process datasets 
               opt =  {'mtype'       : 'bc' or 'met' or 'all'                   | type of metabolites
                       'winsorize'   : 0 or 3 or 6                              | outliers outside of n-iqr are winsozied 
                       'dtype'       : 'cad' or 't2d' or 'all or 'multi'        | disease type, all is mergind cad and t2d, status t2d=1 cad=2 none=0
                       'scaler'      : 'standardize' or 'normalize'             | scaling inputs,  stdr - mean 0 sd 1 // norm - min 0 max 1
                       'cov'         :  predefinedblocks- 'base'/'physical'/'health'/'clinical'/'socio'
                                        or column names in a list 
                                        or 'cov_w_drug' or 'cov_wo_drug' for entire covariate sets with/without drug usage covariates
                       'incidence'   : 'cad' or 't2d'                           | column name of incidence
                       'tte'         : 'tte' or 'year'                          | time-to-event, tte=days / year = years   
                       'exclude'     : a list of column names to be excluded    | example, ['cad_tte','EUR','cad_year']    
                      }

            -> if 'xy_split' is True, split the entire set into 
                X(metabolites+covariates), Incidence(binary index of disease incidence), tte(time-to-event)
        
        < list of datasets >
        met0/3/6 : NMR metabolites, outliers handled outside N-IQR
        bc0/3/6 : Blood Cell Count and Chemistry, outliers handled outside N-IQR
        cov_w/wo_drug : covaraites with or without drug usage columns
        norelated : participants relatedness calculated from genetic PCA 
        cad : Coronary Artery Disease, 
        t2d : Type 2 Diabetes
        t2d-cad(-raw) : merged file, t2d-cad-> merged and processed / t2d-cad-raw -> raw merged 
        disease : multi-output 만들어야됌 
        t2d-cad 
        
        '''
        # load saved dictionary type data set 
        self.root     = root
        self.datapath = datapath # original_datadict2

        # below items are for DataLoader process
        self.inputs   = None
        self.incid    = None
        self.tte      = None 
        self.dset     = torch.load(self.root + self.datapath)
        self.data     = None
        self.xy_split = False
        # dlist is used only for dtype = multi_disease 
        self.dlist    = ['Dementia','MACE','t2d','Liver_Disease','Renal_Disease','Atrial_Fibrillation','Heart_Failure','CHD','Venous_Thrombosis','Cerebral_Stroke',
                        'AAA','PAD','Asthma','COPD','Lung_Cancer','Skin_Cancer','Colon_Cancer','Rectal_Cancer','Parkinson','Fractures','Cataracts','Glaucoma']
        
        # t2d in dlist is from the original t2d data (filtered with multiple sources)
        # other diseases are replicated from NatureMedicine paper using ICD10 only 

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,idx):
        if self.xy_split == False:
            raise ValueError("Run load_data method with 'xy_split = True' ")
        
        inputs   = self.inputs[idx] 
        incid    = self.incid[idx]
        tte      = self.tte[idx]
        return inputs, incid, tte


    # generating a preprocessed dataset merging multiple data sources 
    def load_data(self,opt, xy_split ):
        #load_data(self,mtype,process, dtype, standardize, cov )
        mtype            = opt['mtype']
        winsorize        = opt['winsorize']
        dtype            = opt['dtype']
        self.howtomerge  = opt['howtomerge']
        scaler           = opt['scaler']
        cov              = opt['cov']
        self.xy_split    = xy_split

        # scaler를 데이터셋 여러개 합치고 하면, 샘플수가 달라지고 필터링 되서 분포가 너무 크게 변함 미리 scaling 해둔것을 이용해야함 
        self.data = self._base_dataset(mtype,winsorize, dtype, scaler)

        # add selected covariates 
        if cov is not None: 
            self.data = self._add_cov( data =self.data, cov = cov)

        # split input ,output, excluding columns, if xy_split=True, rerturn tensor of x, y, tte seperately  
        return self.input_output_split(data = self.data, opt = opt, xy_split = self.xy_split)


    # add dataset to datadict 
    def add_dataset(self, datadict, datapath=None):
        # check if the new data set names are in the current dataset 
        if type(datadict) != dict:
            raise ValueError('datadict must me a dictionary, { Name : NewDataSet }')
        ## datadict 
        # if there are overlapped names in the current data dict 
        if len(self.dset.keys() & datadict.keys()):
            print('Datasets, %s, are already in the datadict.' % str(self.dset.keys() & datadict.keys()) )
            if click.confirm('Do you want to overwrite the datasets?', default=True):  
                self.dset.update(datadict)
            else:
                ValueError('Choose different names for the new data sets')

        # if there is no overlapped names in the new datadict 
        else: 
            self.dset.update(datadict)

        ## datapath 
        if (datapath is None) or (datapath == self.datapath):
            print('You are saving the new datadict to the same directory of the current datadict, %s' % str(self.root + self.datapath))
            if click.confirm('Do you want to overwrite the datadict?', default=True):
                newpath = self.root + self.datapath
        else :
            newpath = self.root + datapath

        torch.save(self.dset, newpath)
        print('New datasets are added to the datadict and saved at %s'% newpath)
           
    # remove dataset from datadict 
    def remove_dataset(self, names, datapath=None ):
        if type(names) is not list:
            raise ValueError('name must be a list')
        if len( names & self.dset.keys()):
            print('You are removing data %s from the datadict.'% str(names & self.dset.keys()))
            if click.confirm('Do you want to remove %s from the datadict?' % str(names & self.dset.keys()), default=True):
                for name in names:
                    del self.dset[name]

                if datapath is None: 
                    newpath = self.root+self.datapath
                else:
                    newpath = self.root+datapath
                torch.save(self.dset, newpath)
        else:
            print('The dataset is not in the datadict.')

    # function that manipulates data for training 
    def input_output_split(self, data, opt,  xy_split):

        # dlist is used only when dtype = 'multi_disease'
      
        data       = data
        exclude    = opt['exclude']
        incidence  = self.dlist if opt['dtype'] == 'multi_disease' else opt['incidence']
        tte        = [d+'_tte' for d in self.dlist] if opt['dtype'] == 'multi_disease' else opt['tte']
        excols     = exclude + ['f.eid', incidence, tte]

        # arrange column orders , f.eid / incid / tte / inputX 
        if isinstance(tte, list):
            d1 = data[['f.eid']+tte]
            d2 = data.loc[:,~data.columns.isin(tte)]
            data = pd.merge(d1,d2)

        else:
            data.insert(0, tte, data.pop(tte))
        
        if isinstance(incidence, list):
            d1 = data[['f.eid']+incidence]
            d2 = data.loc[:,~data.columns.isin(incidence)]
            data = pd.merge(d1,d2)
        else: 
            data.insert(0, incidence, data.pop(incidence))

        data.insert(0, 'f.eid', data.pop('f.eid'))

        self.data = data.loc[:, ~data.columns.isin(exclude)]
        '''
        # print balanced class weight for training 
        if opt['dtype'] != 'multi_disease':

            dd = self.data.loc[~self.data[incidence].isin([100]),:]
            bal_weight = list(np.round(compute_class_weight(class_weight = "balanced" ,classes = np.unique(dd[incidence]),y= dd[incidence]),3))
            print('Suggested balanced label weight: {0}'.format(bal_weight))

        else: 
            d2 = [d for d in self.dlist if d not in excols]
            for d in d2:

                dd = self.data[['f.eid',d]].loc[~self.data[['f.eid',d]][d].isin([100]),:]
                bal_weight = list(np.round(compute_class_weight(class_weight = "balanced" ,classes = np.unique(dd[d]),y= dd[d]),3))
                print('Suggested balanced label weight for {0}: {1}'.format(d,bal_weight))

        '''
        if self.xy_split:
            self.inputs  = torch.tensor(self.data.loc[:, ~data.columns.isin(excols)].astype("float32").to_numpy())
            self.incid  = torch.tensor(self.data.loc[:, incidence].astype("int64").to_numpy())
            self.tte    = torch.tensor(self.data.loc[:, tte].astype("float32").to_numpy())

            return (self.inputs, self.incid, self.tte)


        else: 
            return self.data 
    

    ## below codes are supplementary functions for load_data 
    ### functions for generating combined data set 
    def _base_dataset(self, mtype, winsorize, dtype, scaler):
        if mtype not in {'bc','met','all'}:
            raise ValueError('Wrong mtype input, choose bc, met or all')
        if winsorize not in {0,3,6}:
            raise ValueError('Wrong winsorization input, choose 0,3 or 6')
        if dtype not in {'cad', 't2d', 't2d-cad', 'multi_disease'}:
            raise ValueError('Wrong dtype input, choose cad, t2d, t2d-cad or multi_disease')
        if scaler not in {'standardize', 'normalize', None}:
            raise ValueError('Wrong scaler type, choose standardize, normalize or None')
        
        # merge Metabolite set(s) And Disease set 
        # scale data first then merge them 
        if mtype =='all':
            m = [self.dset['bc'+str(winsorize)], self.dset['met'+str(winsorize)]]
        else: 
            m = [self.dset[str(mtype) + str(winsorize)]]

        if scaler is not None: 
            m = [self._Scaler(data = x,type = scaler, exclude = ['f.eid']) for x in m]
        
        m = m.dropna()
        # merge metabolite data and disease data 
        d = [self.dset[dtype]]
        out = self._mergedataset(m+d,MandD = True)  
        
        return out 

    def _mergedataset(self, datalist, MandD = False):
        # merge data sets on f.eid, if MandD True, drop rows EUR != 1 
        # MandD = merge and drop
        datalist.append(self.dset['norelated'])
        output = ft.reduce(lambda left, right: pd.merge(left, right, on='f.eid', how = self.howtomerge), datalist).reset_index(drop=True)
        if MandD: 
            output = output.loc[output['EUR']==1,:].drop(['EUR'],axis=1).reset_index(drop=True)

        return output

    def _add_cov(self, data, cov):
        # select pre-defined covariates or type in self-defined list of covariate columns 
        # covars are list 
        if type(cov) == list:
            #pre-defined covaraites groups 
            cov_dict = {'base'              : ['age','center', 'gender','PC1','PC2','PC3','PC4'],
                        'physical'         : ['bmi','waist_hip_ratio', 'standing_height', 'weight', 'sbp', 'dbp'],
                        'health'            : ['current_smk','log_alcohol_weekly_g'],
                        'clinical'          : ['htn', 'lipid_lowering_med','bp_lowering_med'],
                        'family_history'    : ['father_history_HeartDisease', 'father_history_Stroke','father_history_LungCancer', 'father_history_BowelCancer',
                                               'father_history_BreastCancer', 'father_history_Emphysema','father_history_HighBloodPressure', 'father_history_Diabetes',
                                               'father_history_Alzheimer', 'father_history_Parkinson','father_history_Depression', 'father_history_ProstateCancer',
                                               'mother_history_HeartDisease', 'mother_history_Stroke','mother_history_LungCancer', 'mother_history_BowelCancer',
                                               'mother_history_BreastCancer', 'mother_history_Emphysema','mother_history_HighBloodPressure', 'mother_history_Diabetes',
                                               'mother_history_Alzheimer', 'mother_history_Parkinson','mother_history_Depression'],
                        'socio'             : ['edu_yrs','tdi'],# 'college' excluded, collinearity with edu_yrs
                        'categorical_covs' : ['center']} # this one only used to convert categorical covariates into one-hot encoding
            

            # in the loop, find if the given covariates are in the predefined cov_dict
            cols = ['f.eid']
            for c in cov:
                if c in cov_dict.keys():
                    cols = cols + cov_dict[c]
                else:
                    cols.append(c)

            if len(cols) == 1: # no other 
                raise ValueError('Invalid column name list is given, please check the saved data in MetabDataset.datadict() or "cov_w_drug" or "cov_wo_drug" columns ')
            cols = list(set(cols))  # remove repeated  column names 
            
            covariates = self.dset['cov_w_drug'] # 'cov_w_drug' is the entire covaraites set, select columns only in the given 'cov' list 
            covs = covariates[cols]

        else:
            # all covraites in 'cov_w_drug' or 'cov_wo_drug'
            if cov not in { 'cov_w_drug', 'cov_wo_drug'}:
                raise ValueError('Invalid column name list is given, please check the saved data in MetabDataset.datadict() or use "cov_w_drug" or "cov_wo_drug" ')

            covaraites = self.dset[cov]
            covs = covariates.loc[:, ~covaraites.columns.isin('blood_time','blood_time2','blood_time3','EUR')]
        covs = covs.dropna()
        
        # replace categorial columns into one-hot encoding 
        try: 
            for cat in cov_dict['categorical_covs']:
                covs = pd.concat([covs, pd.get_dummies(covs[cat])],axis=1).drop(columns=cat)
        except: 
                pass
        dat = self._mergedataset([data, covs], MandD =False)
        return dat

    def _Scaler(self, data, type = 'standardize', exclude = None): 
        # scale data with mean 0, sd = 1 
        if exclude is not None:
            ex = data.loc[:, exclude]
            data = data.loc[:, ~data.columns.isin(exclude) ]
         
        x = data.values 
        colname = data.columns 
        if type == 'normalize':
            scaler = preprocessing.MinMaxScaler()
        elif type == 'standardize':
            scaler = preprocessing.StandardScaler()


        x_scaled = scaler.fit_transform(x)
        output = pd.DataFrame(x_scaled, columns =colname)
        
        if exclude is not None:
            output = pd.concat([ex, output], axis = 1)
        
        return output 



class MetabDataLoader(Dataset):
    def __init__(self, opt):
        super().__init__()
        # ratio = [train, test, val], sum = 1
        self.data            = opt['data']
        self.ratio           = opt['ratio']
        self.seed            = opt['seed']
        self.len             = len(self.data)
        self.batch_size      = opt['batch_size']
        self.inference_batch = opt['inference_batch']

        crit = sum(self.ratio)
        if (crit > 1) or (crit < 1-1.0e-5):
            raise ValueError('train, test, validation ratio must sum to 1')
        
        if (self.ratio[0] == 0) or (self.ratio[1] == 0):
            raise ValueError('train and val cannot be zero')
    
        train_size = int(self.ratio[0] * self.len)
        val_size  = int(self.ratio[2] * self.len)
        test_size   = self.len - train_size - val_size
        # data split index 
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        self.train, self.test, self.val = random_split(self.data, [train_size, test_size, val_size], generator = gen)

        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
       

    def data_loader(self):
        # shuffle here is for shuffle data inside train / test set
        # shuffle in __init__ for shuffle index from original data 
        # test/val batch is full length, thus only one batch is given by default
        # inference batch can be used when the GPU memory exceeded
        
        test_batch = len(self.test) if self.inference_batch == None else self.inference_batch
        val_batch =  len(self.val)  if self.inference_batch == None else self.inference_batch

        self.train_loader = DataLoader(dataset = self.train, batch_size = self.batch_size , shuffle = True)
        self.val_loader   = DataLoader(dataset = self.val,   batch_size = val_batch,   shuffle = False)
        self.test_loader  = DataLoader(dataset = self.test,  batch_size = test_batch,  shuffle = False)
        
        return self.train_loader,self.val_loader,self.test_loader


    

