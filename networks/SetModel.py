from .modules import *
import torch.nn.functional as F

class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output))

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=5, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))


class SetTransformer4Anomaly(nn.Module):
    def __init__(self, dim_input, dim_output,
            num_inds=5, dim_hidden=1280, num_heads=4, ln=False):
        super(SetTransformer4Anomaly, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_output, num_heads, num_inds, ln=ln))
        
        '''
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))
        '''
        #self.fc = nn.Linear(dim_hidden*num_heads*pos_num, pos_num)

    def forward(self, X):
        embed =  self.enc(X)
        embed = embed.view(embed.size(0), -1)
        return embed

# my implementation of https://arxiv.org/abs/1802.04712

class MaxWelling(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(MaxWelling, self).__init__()
        self.att_weight = nn.Sequential(
                nn.BatchNorm1d(dim_input),
                nn.ReLU(True),
                nn.Linear(dim_input, 1)
            )
        self.lin = nn.Sequential(
                nn.BatchNorm1d(dim_input),
                nn.ReLU(True),
                nn.Linear(dim_input, dim_output)
                )

    def forward(self, x):
        instance_num = x.size(1)
        x_size = x.size()
        weight_res = []

        for i in range(instance_num):
            x_med = x[:,i,:]
            x_med = x_med.view((x_size[0], -1))
            weight = self.att_weight(x_med)
            weight_res.append(weight)

        embed_weight = torch.cat(weight_res, 1)
        embed_weight = F.softmax(embed_weight, dim=1)

        embed_weight = torch.reshape(embed_weight, (embed_weight.size(0), 1, -1))
        rep = torch.bmm(embed_weight, x)
        rep = torch.reshape(rep, (rep.size(0), -1))

        out = self.lin(rep)

        return out


