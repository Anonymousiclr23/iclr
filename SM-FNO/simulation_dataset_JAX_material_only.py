import os.path
import numpy as np
import random
from torch.utils.data import Dataset

from datetime import datetime

class SimulationDataset(Dataset):
    def __init__(self, data_folder, total_sample_number = None, transform = None, data_mult = 1):

        self.yeex = np.load(data_folder+'/'+'cropped_yeex.npy', mmap_mode='r')
        self.yeex = self.yeex[:, :, :].astype(np.float32)
        # sub = np.ones((self.yeex.shape[0],1,5,256))*(1.45-1.)/(3.567390909090909-1.)
        # self.yeex=np.concatenate([sub,self.yeex],axis=2).astype(np.float32);
        print("yeex.shape: ", self.yeex.shape, self.yeex.dtype)

        self.yeey = np.load(data_folder+'/'+'cropped_yeey.npy', mmap_mode='r')
        self.yeey = self.yeey[:, :, :].astype(np.float32)
        print("yeey.shape: ", self.yeey.shape, self.yeey.dtype)

        # # data_folder = config.get('data_folder', '/scratch/users/chenkaim/data/') # for sherlock
        # self.input_imgs = np.load(data_folder+'/'+'cropped_imgs.npy', mmap_mode='r') 
        # self.input_imgs = self.input_imgs[:, :, :, None].astype(np.float32)
        # self.input_imgs = np.transpose(self.input_imgs, 1, 3);
        # # sub = np.ones((self.input_imgs.shape[0],1,5,256))*(1.45-1.)/(3.567390909090909-1.)
        # # self.input_imgs=np.concatenate([sub,self.input_imgs],axis=2).astype(np.float32);
        # print("input_imgs.shape: ", self.input_imgs.shape, self.input_imgs.dtype)
        
        self.Hy_forward = np.load(data_folder+'/'+'cropped_Hys.npy', mmap_mode='r');
        self.Hy_forward = data_mult*np.stack((np.real(self.Hy_forward),np.imag(self.Hy_forward)),axis=3).astype(np.float32);
        print("Hy_forward.shape: ", self.Hy_forward.shape, self.Hy_forward.dtype)
        

        self.top_bc = np.load(data_folder+'/'+'cropped_top_bc.npy', mmap_mode='r');
        self.top_bc = data_mult*np.stack((np.real(self.top_bc),np.imag(self.top_bc)),axis=3).astype(np.float32);
        print("top_bc.shape: ", self.top_bc.shape, self.top_bc.dtype)
        
        self.bottom_bc = np.load(data_folder+'/'+'cropped_bottom_bc.npy', mmap_mode='r');
        self.bottom_bc = data_mult*np.stack((np.real(self.bottom_bc),np.imag(self.bottom_bc)),axis=3).astype(np.float32);
        print("bottom_bc.shape: ", self.bottom_bc.shape, self.bottom_bc.dtype)

        self.left_bc = np.load(data_folder+'/'+'cropped_left_bc.npy', mmap_mode='r');
        self.left_bc = data_mult*np.stack((np.real(self.left_bc),np.imag(self.left_bc)),axis=3).astype(np.float32);
        print("left_bc.shape: ", self.left_bc.shape, self.left_bc.dtype)

        self.right_bc = np.load(data_folder+'/'+'cropped_right_bc.npy', mmap_mode='r');
        self.right_bc = data_mult*np.stack((np.real(self.right_bc),np.imag(self.right_bc)),axis=3).astype(np.float32);
        print("right_bc.shape: ", self.right_bc.shape, self.right_bc.dtype)

        self.fields = self.Hy_forward;
        
        if total_sample_number:
            random.seed(1234)
            indices = np.array(random.sample(list(range(self.Hy_forward.shape[0])), total_sample_number))
            self.yeex = np.take(self.yeex, indices, axis=0)
            self.yeey = np.take(self.yeey, indices, axis=0)
            # self.input_imgs = np.take(self.input_imgs, indices, axis=0)
            self.fields = np.take(self.fields, indices, axis=0)
            self.top_bc = np.take(self.top_bc, indices, axis=0)
            self.bottom_bc = np.take(self.bottom_bc, indices, axis=0)
            self.left_bc = np.take(self.left_bc, indices, axis=0)
            self.right_bc = np.take(self.right_bc, indices, axis=0)
            print("finished indexing")

        self.transform = transform

    def __len__(self):
        return self.fields.shape[0]

    def __getitem__(self, idx):
        # structure = self.input_imgs[idx, :, :, :]
        field = self.fields[idx, :, :, :]
        top_bc = self.top_bc[idx, :, :, :]
        bottom_bc = self.bottom_bc[idx, :, :, :]
        left_bc = self.left_bc[idx, :, :, :]
        right_bc = self.right_bc[idx, :, :, :]
        yeex = self.yeex[idx, :, :]
        yeey = self.yeey[idx, :, :]

        #means = [self.Hy_meanR, self.Hy_meanI, self.Ex_meanR, self.Ex_meanI, self.Ez_meanR, self.Ez_meanI];

        sample = {'field': field, 
                  'top_bc': top_bc, 'bottom_bc': bottom_bc,
                  'left_bc': left_bc, 'right_bc': right_bc,
                  'yeex': yeex, 'yeey': yeey}#, 'means': means};

        if self.transform:
            sample = self.transform(sample)

        return sample
