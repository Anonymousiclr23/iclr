import os.path
import numpy as np
import random
from torch.utils.data import Dataset

from datetime import datetime

class SimulationDataset(Dataset):
    def __init__(self, data_folder, total_sample_number = None, transform = None, data_mult = 1):
        self.Hy_forward = np.load(data_folder+'/'+'cropped_Hys.npy', mmap_mode='r');
        self.Hy_forward = data_mult*np.stack((np.real(self.Hy_forward),np.imag(self.Hy_forward)),axis=3).astype(np.float32);
        print("Hy_forward.shape: ", self.Hy_forward.shape, self.Hy_forward.dtype)

        self.sources = np.load(data_folder+'/'+'cropped_sources.npy', mmap_mode='r');
        self.sources = data_mult*np.stack((np.real(self.sources),np.imag(self.sources)),axis=3).astype(np.float32);
        print("sources.shape: ", self.sources.shape, self.sources.dtype)


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
            self.fields = np.take(self.fields, indices, axis=0)
            self.sources = np.take(self.sources, indices, axis=0)

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
        source = self.sources[idx, :, :, :]
        

        #means = [self.Hy_meanR, self.Hy_meanI, self.Ex_meanR, self.Ex_meanI, self.Ez_meanR, self.Ez_meanI];

        sample = {'field': field, 'source': source,
                  'top_bc': top_bc, 'bottom_bc': bottom_bc,
                  'left_bc': left_bc, 'right_bc': right_bc}#, 'means': means};

        if self.transform:
            sample = self.transform(sample)


        return sample
