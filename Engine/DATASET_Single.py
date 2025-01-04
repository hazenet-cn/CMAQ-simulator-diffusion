import os
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(
        self, 
        args
    ):
        super(CustomDataset, self).__init__()
        assert args.mode in ['train', 'sampling'], 'mode must be train or sampling.'
        self.window = args.window
        self.width = args.width
        self.length = args.length
        self.layers = args.layers
        self.data = np.load(args.input_dir)
        if args.mode == 'train':
            self.data = self.data[:args.train_timestep,:]
        if args.mode == 'sampling':
            self.data = self.data[-25:,:]
        self.data, self.scaler = self.Norm(self.data)
        self.data = np.concatenate((self.data[:-1,:],self.data[1:,:self.layers,:]), axis=1)
        #if args.mode == 'train' and args.train_timestep >= 3909 :
            #self.data = np.delete(self.data, 3909, axis=0)  # Missing data for June
         
    def Norm(self, data):
        a,b,c = data.shape[1:]
        timestep = data.shape[0]
        aconc = data[:,:self.layers,:,:]
        other = data[:,self.layers:,:,:]

        other = other.reshape(timestep,(a-self.layers)*b*c)
        scaler = StandardScaler()
        scaler = scaler.fit(other)
        other = scaler.transform(other)
        other = other.reshape(timestep,(a-self.layers),b,c)

        aconc = aconc.reshape(timestep,self.layers*b*c)  
        scaler = StandardScaler()
        scaler = scaler.fit(aconc)
        aconc = scaler.transform(aconc)
        aconc = aconc.reshape(timestep,self.layers,b,c)
    
        data = np.concatenate((aconc[:,:self.layers,:,:], other), axis=1)
        data = np.concatenate((data,aconc[:,self.layers:,:,:]), axis=1)
        return data, scaler




