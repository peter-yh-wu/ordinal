'''
MLP supporting ordinal triplet loss implementation

Peter Wu
peterw1@andrew.cmu.edu
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_TriSlab(nn.Module):
    '''MLP supporting ordinal triplet loss
    
    Halves hidden dimension each subsequent layer
    '''
    def __init__(self, input_dim, output_dim, args):
        super(MLP_TriSlab, self).__init__()
        hidden_dim = args.hidden_dim
        num_layers = args.num_layers
        dropout = args.dropout
        self.emb_dim = args.emb_dim

        hidden_dims = [hidden_dim for _ in range(num_layers)]
        
        log_input_dim = int(math.log(input_dim, 2))
        log_output_dim = int(math.log(self.emb_dim, 2))
        delta = (log_input_dim-log_output_dim)/(num_layers+1)
        log_hidden_dims = [log_input_dim-delta*(i+1) for i in range(num_layers)]
        hidden_dims = [int(math.pow(2, l)) for l in log_hidden_dims]
    
        dims = [input_dim]+hidden_dims
        self.fc_layers = nn.ModuleList([
                nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.Dropout(dropout), nn.ReLU()) \
            for i in range(num_layers)])
        self.emb_output = nn.Sequential(nn.Linear(dims[-1], self.emb_dim), nn.BatchNorm1d(self.emb_dim))
        fc2_dim = int(self.emb_dim/2)
        self.output = nn.Sequential(nn.ReLU(), nn.Linear(self.emb_dim, fc2_dim), nn.ReLU(), nn.Linear(fc2_dim, output_dim))

    def forward_emb(self, x):
        for i, l in enumerate(self.fc_layers):
            x = self.fc_layers[i](x)
        x = self.emb_output(x)
        return x

    def forward_slab(self, x):
        x = self.output(x)
        x = F.softmax(x, 1)
        return x

    def forward(self, x):
        for i, l in enumerate(self.fc_layers):
            x = self.fc_layers[i](x)
        x = self.emb_output(x)
        x = self.output(x)
        return x
