import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from einops import rearrange
import math
from scipy.io import loadmat

class TrainDataset(Dataset):
    def __init__(self, transform=None):
        super().__init__()
        
        self.path_log = "/srv/csj/tutorial_submit/shuffle.txt"

        # In shuffle.txt, the first 90% are for training, the last 10% are for testing

        self.path_list = []

        with open(self.path_log, 'r') as txt:
            for line in txt:
                self.path_list.append(line[:-1])

        self.train_len = math.floor(len(self.path_list)*0.9)

        self.path_list = self.path_list[:self.train_len]

        if transform:
            self.transform = transform
        else:
            self.transform = self.transformers_preprocess

    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, index):
        path = self.path_list[index]

        tensor = self.get_mat(path)
        if self.transform:
            tensor = self.transform(tensor)

        label = self.get_label(path)

        return tensor, label
    
    def get_label(self, path):
        path = path.split("/")[-1][:-4]
        id, a, b, c, d, Rx = path.split("-")
        label = int(a)-1
        return label
    
            
    def get_mat(self, path):
        array = loadmat(path)['cfr_array']
        tensor = torch.from_numpy(array)
        tensor = tensor.to(torch.complex64)
        return tensor

    def transformers_preprocess(self, x):

        real_part = torch.real(x)
        real_part = rearrange(real_part, "l d -> 1 d l")
        real_part = F.interpolate(real_part, size=1000, mode='nearest-exact')
        real_part = rearrange(real_part, "1 d l -> l d")

        imag_part = torch.imag(x)
        imag_part = rearrange(imag_part, "l d -> 1 d l")
        imag_part = F.interpolate(imag_part, size=1000, mode='nearest-exact')
        imag_part = rearrange(imag_part, "1 d l -> l d")

        complex_tensor = torch.complex(real_part ,imag_part)

        return complex_tensor
    


