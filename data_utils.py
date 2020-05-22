import torch.utils.data as data
import torch
import numpy as np

from os import listdir
from os.path import join
import scipy.io as scio


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])
               
   
class TrainsetFromFolder(data.Dataset):
    def __init__(self, dataset_dir):
        super(TrainsetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir , x) for x in listdir(dataset_dir ) if is_image_file(x)]

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index], verify_compressed_data_integrity=False)
        input = mat['lr'].astype(np.float32)
        label = mat['hr'].astype(np.float32)
        
        return torch.from_numpy(input), torch.from_numpy(label)
        
    def __len__(self):
        return len(self.image_filenames)        
        
     
class ValsetFromFolder(data.Dataset):
    def __init__(self, dataset_dir):
        super(ValsetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir , x) for x in listdir(dataset_dir ) if is_image_file(x)]

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index])
        input = mat['LR'].astype(np.float32).transpose(2, 0, 1)
        label = mat['HR'].astype(np.float32).transpose(2, 0, 1)
             	                  
        
        return torch.from_numpy(input).float(), torch.from_numpy(label).float()


    def __len__(self):
        return len(self.image_filenames)   
