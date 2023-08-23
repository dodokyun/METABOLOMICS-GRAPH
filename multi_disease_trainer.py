import pandas as pd 
import numpy as np 
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as gDataLoader
import torch_geometric
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import  f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from collections import OrderedDict

class trainer_multidisease():
    def __init__(self, model, graph_list, optimizer,split_ratio,num_classes,device,ordered_tasks,mask=100,
                 seed = 12345, batch_size=16, epoch = 100,path=None,inference_batch=None):
        super().__init__()
        # model: gnn model
        # graph_list : torch_geometric graph data in a list
        # split_ratio : [train, test, val] a list of ratio sum to 1 
        # num_classes = classification class 
        # seed : shuffle random seed 
        # batch_size : training batch size
        # inference_batch = batch size for val and test set, if none -> entire batch for val and test 
    
        self.device = device
        self.model = model.to(self.device)
        self.graph_list = graph_list
        self.ordered_tasks = ordered_tasks # loss function dictionary with distinctive loss weights
        self.criterion = self.loss_functions()
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=0.005)
        self.num_classes = num_classes 
        self.epoch = epoch
        self.seed = seed  
        self.path = path

        self.mask_indicator = torch.tensor([mask]).to(self.device) 
      
        # data splitter with random seed 
        self.len = len(graph_list)
        self.train_loader, self.test_loader, self.val_loader = self.data_loader(split_ratio, batch_size, inference_batch)


    def forward(self):
        for ep in range(self.epoch):
            # train and validation 
            train_loss, self.train_out_dict, self.train_label_dict = self.training_step()
            self.val_out_dict, self.val_label_dict = self.evals(dataset = self.val_loader)

            # print reuslt metrics 
            #if ep %10 == 1:
            print(f'< Epoch: {ep} >', flush =True)
            print('Train loss : {0:.3f}'.format(train_loss), flush=True) 
            print('{0} classses prediction'.format(self.num_classes))
            
            try:
                print(' <Training Metrics> ')
                self.train_metric_dict = {task : self.calculate_metric(y_pred = self.train_out_dict[task], y_true=self.train_label_dict[task],task=task, tr='Train') for task in self.ordered_tasks}
            except:
                print('Error occurs while calculating training metrics')
            try:
                print('< Validation Metrics >')
                self.val_metric_dict = {task : self.calculate_metric(y_pred = self.val_out_dict[task], y_true=self.val_label_dict[task],task=task, tr='Val') for task in self.ordered_tasks}
            except: 
                print('Error occurs while calculating validation metrics')

            if self.path is not None:

                for task in self.ordered_tasks:
                    if not os.path.exists(self.path):
                        os.mkdir(path)

                    if not os.path.exists(f'{self.path}/{task}_log.csv'):
                        with open(f'{self.path}_auclog.csv', 'w') as file:
                            file.write('tr_accuracy,tr_precision,tr_recall,tr_f1,tr_auc,val_accuracy,val_precision,val_recall,val_f1,val_auc\n')
                        
                    else:
                        train_task_mtr = train_metric_dict[task]
                        with open(f'{self.path}/{task}_log.csv', 'a') as file:
                            file.write('{0:.3f},{1:.3f},{2:.3f},{3:.3f},{4:.3f},{5:.3f},{6:.3f},{7:.3f},{8:.3f},{9:.3f}\n'.format(train_task_mtr[0],train_task_mtr[1],train_task_mtr[2],train_task_mtr[3],train_task_mtr[4],self._val_mtr[0],self._val_mtr[1],self._val_mtr[2],self._val_mtr[3],self._val_mtr[4]))
                                
                if ep%30 ==0:
                    model.sharedGNNblock.state_dict()
                    torch.save(model.sharedGNNblock.state_dict(), self.path+'/sharedGNN.pt')
                    torch.save(self.model.state_dict(), self.path+'/entireModel.pt')
                break

      
    def training_step(self):
        total_loss = 0 
        torch.cuda.empty_cache()
    
        train_out_dict = {t :[] for t in self.ordered_tasks}
        train_label_dict = {t :[] for t in self.ordered_tasks}

        self.model.train()
        for batch in self.train_loader:
            batch = batch.to(self.device) 
            label = batch.y # dictionary 

            ## 여기에서 batch단위로 나온 결과들 epoch단위로 합쳐서 아웃풋 내보내야 auc 계산할수있으니까 이거 고쳐
            self.optimizer.zero_grad()
            out = self.model(batch) # out : dictinonary ex) {T2D: [[0.05, 0.95], [0.80,0.20],[0.4,0.5]]}
            out,label = self.task_specific_mask(out, label) # label_dict ex) {T2D: [1,0,0]}
            loss = self.loss(out, label)
            loss.backward()
            total_loss += loss.item() * batch.num_graphs
            self.optimizer.step()
            train_out_dict,train_label_dict = self.prediction_repository(out, label,train_out_dict, train_label_dict )

        return total_loss, train_out_dict, train_label_dict
    
    def evals(self, dataset):
        torch.cuda.empty_cache()
        eval_out_dict = {t :[] for t in self.ordered_tasks}
        eval_label_dict = {t :[] for t in self.ordered_tasks}

        self.model.eval()
        with torch.no_grad():
            for data in dataset:
                data = data.to(self.device)
                label = data.y
                out = self.model(data)
                out,label = self.task_specific_mask(out, label)
                eval_out_dict,eval_label_dict = self.prediction_repository(out, label,eval_out_dict, eval_label_dict )

            return eval_out_dict, eval_label_dict


    def model_test_result(self,tasks=None, dataset=None):
        if dataset is None:
            dataset = self.test_loader

        eval_out_dict, eval_label_dict = self.evals(dataset)
        eval_metric_dict = {task : self.calculate_metric(eval_out_dict[task], eval_label_dict[task],task, tr='Test') for task in self.ordered_tasks}

        return eval_metric_dict

    def task_specific_mask(self, out,label):
        # mask는 label을 기반으로, if label == 100 -> mask!
        #label = label.reshape(-1,len(self.ordered_tasks))

        mask = {key: ~torch.eq(value, self.mask_indicator) for key, value in label.items()}
        #mask = ~torch.eq(label, self.mask_indicator)       
        #label_dict = {t :[] for t in self.ordered_tasks}
        
        for i, task in enumerate(self.ordered_tasks):
            # each column of mask represents mask indicator of each disease
            #m = mask[:,i] # masking for single disease

            m = mask[task]
            out[task] = out[task][m]
            label[task] = label[task][m]

            # task in dict = columns of label -> masking -> only valid labels are to be saved
            #label_dict[task] = label[:,i][m]

        return out, label#label_dict


    def loss(self, out, label):
        loss = 0 
        for task in out.keys():
            lossfx =self.criterion[task]
            head_loss = lossfx(out[task], label[task])
            loss += head_loss

        return loss

    def loss_functions(self):
        # belows are weights for balanced label, [0,1]
        weight_per_task = {'Dementia':[0.5,45],
                'MACE':[0.5,7.2],
                'Liver_Disease':[0.5,13.9],
                'Renal_Disease':[0.5,8.4],
                'Atrial_Fibrillation' :[0.5,10.5],
                'Heart_Failure':[0.5,28.7],
                'CHD':[0.5,8],
                'Venous_Thrombosis':[0.5,47.6],
                'Cerebral_Stroke':[0.5,38.1],
                'AAA':[0.5,96.8],
                'PAD':[0.5,22.9],
                'Asthma':[0.5,8.7],
                'COPD':[0.5,15.2],
                'Lung_Cancer':[0.5,60.9],
                'Skin_Cancer':[0.5,14.8],
                'Colon_Cancer':[0.5,56.2],
                'Rectal_Cancer':[0.5,100.5],
                'Parkinson':[0.5,86.6],
               'Fractures':[0.5,10],
               'Cataracts':[0.5,6.5],
               'Glaucoma':[0.5,27.5],
               't2d':[0.5,12]
               }
        lossfxs = { task: torch.nn.CrossEntropyLoss(weight= torch.tensor(weight_per_task[task],dtype=torch.float).to(device)) for task in self.ordered_tasks}

        return lossfxs


    def prediction_repository(self, out, label, train_out_dict, train_label_dict):
        for task in out.keys():
            train_out_dict[task] = np.append(train_out_dict[task], out[task].argmax(dim=1).detach().cpu().numpy())
            train_label_dict[task] = np.append(train_label_dict[task], label[task].detach().cpu().numpy())
        return train_out_dict, train_label_dict


    def calculate_metric(self,y_pred, y_true,task, tr ='train'):
        if self.num_classes > 2: 
            acc = accuracy_score(y_true,y_pred)
            pcs = precision_score(y_true,y_pred, average = 'macro')
            rcll = recall_score(y_true,y_pred, average = 'macro')
            f1 = f1_score(y_true,y_pred, average = 'macro')
            
            y_pred = self.get_one_hot(y_pred, self.num_classes)
            auc = roc_auc_score(y_true,y_pred,average = 'macro', multi_class='ovo')
        
        else:
            acc = accuracy_score(y_true,y_pred)
            pcs = precision_score(y_true,y_pred)
            rcll = recall_score(y_true,y_pred)
            f1 = f1_score(y_true,y_pred)
            auc = roc_auc_score(y_true,y_pred)
        
        print("{0} for {1} | Accuray : {2:.2f} / Precision : {3:.2f} / Recall : {4:.2f} / f1-score : {5:.2f} / AUROC :{6:.2f}".format(tr,task, acc, pcs, rcll, f1,auc), flush =True)
        return [acc, pcs, rcll, f1, auc]

    
    def get_one_hot(self, y_pred, n_classes):
        one_hot = np.eye(n_classes)[np.array(y_pred).reshape(-1)]
        return one_hot        


    def data_loader(self, split_ratio, batch_size, inference_batch):
        crit = sum(split_ratio)
        if (crit > 1) or (crit < 1-1.0e-5):
            raise ValueError('train, test, validation ratio must sum to 1')

        if (split_ratio[0] == 0) or (split_ratio[2] == 0):
            raise ValueError('train and val cannot be zero')
    
        train_size = int(split_ratio[0] * self.len)
        val_size   = int(split_ratio[2] * self.len)
        test_size  = self.len - train_size - val_size
       
       
        # data split index 
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        
        train, test, val = random_split(self.graph_list, [train_size, test_size, val_size], generator = gen)
        
        test_batch = len(test) if inference_batch == None else inference_batch
        val_batch =  len(val)  if inference_batch == None else inference_batch
        
        train_loader = gDataLoader(dataset = train, batch_size = batch_size , shuffle = True, drop_last= True)
        test_loader  = gDataLoader(dataset = test,  batch_size = test_batch,  shuffle = False, drop_last= True)
        val_loader   = gDataLoader(dataset = val,   batch_size = val_batch,   shuffle = False, drop_last= True)
        return train_loader, test_loader, val_loader