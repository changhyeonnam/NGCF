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

        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss


    def __call__(self,*args):
        return self.forward(*args)