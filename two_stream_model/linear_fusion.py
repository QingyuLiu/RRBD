from torch import nn
import torch
from copy import deepcopy
class HdmProdBilinearFusion(nn.Module):

    def __init__(self, dim1, dim2,
                 hidden_dim=2048, output_dim=3072, bili_affine=None,
                 bili_dropout=0.5, **kwargs):
        super(HdmProdBilinearFusion, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.Trans1 = nn.Linear(dim1, hidden_dim)
        self.Trans2 = nn.Linear(dim2, hidden_dim)
        self.OutTrans = nn.Linear(hidden_dim, output_dim)

        if bili_dropout is None:
            self.Dropout = nn.Identity()
        else:
            self.Dropout = nn.Dropout(bili_dropout)

    def forward(self, features1, features2, **kwargs):
        b1, c1, h1, w1 = features1.size()
        b2, c2, h2, w2 = features2.size()
        assert b1 == b2 and h1 == h2 and w1 == w1
        features1 = features1.view(b1, c1, -1).permute(0, 2, 1).contiguous().view(-1, c1)
        features2 = features2.view(b2, c2, -1).permute(0, 2, 1).contiguous().view(-1, c2)
        prod = self.Trans1(features1) * self.Trans2(features2)
        prod = torch.tanh(prod)
        prod = self.OutTrans(self.Dropout(prod))
        prob = prod.view(b1, -1, self.output_dim).permute(0, 2, 1).contiguous().view(b1, -1, h1, w1)

        return prob

