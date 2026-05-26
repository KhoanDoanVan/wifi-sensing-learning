import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import math
from scipy.io import loadmat

class TrainDataset(Dataset):
    def __init__(self, transform=None):
        super().__init__()
        
        self.path_log = "/srv/csj/tutorial_dfs_submit/shuffle.txt"

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
        id, a, b, c, d, suffix = path.split("-")
        label = int(a)-1
        return label
    
            
    def get_mat(self, path):
        array = loadmat(path)['doppler_spectrum']
        tensor = torch.from_numpy(array)
        tensor = tensor.to(torch.float32)
        return tensor

    def transformers_preprocess(self, x):
        x = F.interpolate(x, size=1000, mode='nearest-exact')
        return x

