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

        log_prob = nn.LogSigmoid()(pos_scores - neg_scores).sum()

        regularization = self.decay*(users.norm(dim=1).pow(2).sum()
                                     +pos_items.norm(dim=1).pow(2).sum()
                                     +neg_items.norm(dim=1).pow(2).sum())
        loss =  regularization-log_prob
        loss /=self.batch_size
        return loss


    def __call__(self,*args):
        return self.forward(*args)
