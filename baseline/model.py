'''
Baseline MLP implementation

Peter Wu
peterw1@andrew.cmu.edu
'''

import math
import torch
import torch.nn as nn


class MLP(nn.Module):
    '''MLP that halves hidden dimension each subsequent layer
    '''
    def __init__(self, input_dim, output_dim, args):
        super(MLP, self).__init__()
        hidden_dim = args.hidden_dim
        num_layers = args.num_layers
        dropout = args.dropout

        hidden_dims = [hidden_dim for _ in range(num_layers)]
        
        log_input_dim = int(math.log(input_dim, 2))
        log_output_dim = int(math.log(output_dim, 2))
        delta = (log_input_dim-log_output_dim)/(num_layers+1)
        log_hidden_dims = [log_input_dim-delta*(i+1) for i in range(num_layers)]
        hidden_dims = [int(math.pow(2, l)) for l in log_hidden_dims]
    
        dims = [input_dim]+hidden_dims
        self.fc_layers = nn.ModuleList([
                nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.Dropout(dropout), nn.ReLU()) \
            for i in range(num_layers)])
        self.output = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for i, l in enumerate(self.fc_layers):
            x = self.fc_layers[i](x)
        x = self.output(x)
        return x
