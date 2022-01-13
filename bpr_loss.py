import torch
import torch.nn as nn

class BPR_Loss(nn.Module):
    def __init__(self,
                 batch_size:int,
                 decay_ratio:float=1e-5):
        super(BPR_Loss, self).__init__()

        self.batch_size= batch_size
        self.decay = decay_ratio

    def forward(self,users,pos_items,neg_items):

        pos_scores = torch.mul(users,pos_items).sum(dim=1)
        neg_scores =  torch.mul(users,neg_items).sum(dim=1)

        return pos_scores, neg_scores


    def __call__(self,*args):
        return self.forward(*args)
