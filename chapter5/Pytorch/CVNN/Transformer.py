import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from chapter5.Pytorch.CVNN.complex.complex_module import *

class RF_Transformer(nn.Module):
    def __init__(self, num_classes=6, seglen=8, origin_dim=720, dim=720, heads=8, layers=8, dropout=0.0):
        super(RF_Transformer, self).__init__()

        self.num_classes = num_classes
        self.seglen = seglen
        self.encoder_origin_dim = origin_dim
        self.dim = dim
        self.heads = heads
        self.layers = layers
        self.dropout = dropout
        self.encoder = ComplexTransformerEncoder(
                        origin_dim = self.encoder_origin_dim,
                        key_dim = self.dim,
                        query_dim = self.dim,
                        value_dim = self.dim,
                        hidden_dim = self.dim,
                        norm_shape = self.dim,
                        ffn_input_dim = self.dim,
                        ffn_hidden_dim = self.dim,
                        num_heads = self.heads,
                        num_layers = self.layers,
                        dropout = self.dropout)

        self.c2r = ComplexToReal()

        self.mlp = ComplexMLP(in_features = self.dim, out_features = self.dim )

        self.classifier = nn.Sequential(
            nn.Linear(self.dim, self.dim),  
            nn.ReLU(),
            nn.Dropout(self.dropout),        
            nn.Linear(self.dim, self.num_classes)
        )


    def forward(self, x):
        x = torch.stack((torch.real(x), torch.imag(x)), dim=-1)
        x = rearrange(x, "b (x s) d I -> b x (s d) I", s = self.seglen)
        class_token = torch.zeros(x[:,0:1].shape).to(x.device)
        x = torch.cat((class_token, x), dim = 1)
        x = self.encoder(x)
        x = x[:,0]
        x = self.mlp(x)
        x = self.c2r(x)
        x = self.classifier(x)
        return x
