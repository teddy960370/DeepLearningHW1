# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 13:52:55 2022

@author: ted
"""

import pandas as pd
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class BinaryClassification(nn.Module):
    def __init__(self,input_dimension):
        super(BinaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(input_dimension, 64) 
        self.layer_2 = nn.Linear(64, 32)
        self.layer_out = nn.Linear(32, 1) 
        
        self.LeakyReLU = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(32)
        
    def forward(self, inputs):
        x = self.LeakyReLU(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.dropout(x)
        x = self.LeakyReLU(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x
    
class TrainDataset(Dataset):
 
  def __init__(self,inputDataFrame):
 
    x=inputDataFrame.iloc[:,:-1].values
    y=inputDataFrame.iloc[:,-1].values
 
    self.x = torch.tensor(x,dtype=torch.float)
    self.y = torch.tensor(y,dtype=torch.float)
 
  def __len__(self):
    return len(self.y)
   
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]

class TestDataset(Dataset):
 
  def __init__(self,inputDataFrame):
 
    x=inputDataFrame.values
 
    self.x = torch.tensor(x,dtype=torch.float)
 
  def __len__(self):
    return len(self.x)
   
  def __getitem__(self,idx):
    return self.x[idx]

def train_loop(dataloader,model,optimizer):
    num_batches = len(dataloader)
    total_acc = 0
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    for batch, data in enumerate(dataloader):

        #hidden = model.init_hidden(data[0].size(dim=0))
        model.zero_grad()
        optimizer.zero_grad()
        # Compute prediction and loss
        pred = model(data[0])
        loss = loss_fn(pred, data[1].unsqueeze(1))

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_acc += binary_acc(pred, data[1].unsqueeze(1))
    
    print(f"train_acc: {total_acc/num_batches} ")

def vaild_loop(dataloader,model,optimizer):
    num_batches = len(dataloader)
    total_acc = 0
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for batch, data in enumerate(dataloader):

        #hidden = model.init_hidden(data[0].size(dim=0))
        model.zero_grad()
        optimizer.zero_grad()
        # Compute prediction and loss
        pred = model(data[0])
        loss = loss_fn(pred, data[1].unsqueeze(1))

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_acc += binary_acc(pred, data[1].unsqueeze(1))
    
    print(f"vaild_acc: {total_acc/num_batches} ")
    
def test_model(data_loader, model):
    ans = list()
    with torch.no_grad():
        for data in enumerate(data_loader):
            X = data[1]

            output = model(X)
            pred = torch.round(torch.sigmoid(output))
            ans.append(pred)

    return ans

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc
def main():

    path = './data/'
    trainSet = pd.read_csv(path + 'train_processed.csv')
    testSet = pd.read_csv(path + 'test_processed.csv')
    originTestSet = pd.read_csv(path + 'test.csv')
    
    train,vaild = train_test_split(trainSet, test_size=0.3)
    
    # data loader
    train_loader = DataLoader(TrainDataset(train), batch_size = 128, shuffle = False)
    vaild_loader = DataLoader(TrainDataset(vaild), batch_size = 128, shuffle = False)
    test_loader = DataLoader(TestDataset(testSet), batch_size = 1, shuffle = False)
    
    _,input_dimension = testSet.shape
    model = BinaryClassification(input_dimension)
    
    model.train()
    
    optimizer = optim.Adam(model.parameters(), 1e-3, weight_decay=0.01)
    epoch_size = 300
    epoch_pbar = trange(epoch_size, desc="Epoch")
    
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        train_loop(train_loader, model, optimizer)
        # TODO: Evaluation loop - calculate accuracy and save model weights
        vaild_loop(vaild_loader, model, optimizer)
        pass

    model.eval()
    ans = test_model(test_loader,model)
    
    data = pd.DataFrame()
    data['PassengerId'] = originTestSet['PassengerId']
    data['Transported'] = np.array(ans,dtype=bool)
    data.to_csv(path + "pred.csv",index=False)
    
    
if __name__ == "__main__":
    main()