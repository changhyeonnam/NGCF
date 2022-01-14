import torch
import torch.nn as nn
import numpy as np
class Evaluation():
    def __init__(self,
                 test_dataloader,
                 model,
                 top_k:int=5):
        self.dataloader = test_dataloader
        self.model = model
        self.top_k = top_k

    def dcg(self,gt_items):
        dcg=[]
        for idx,item in enumerate(gt_items):
            dcg.append(item/np.log2(idx+1))
        return np.sum(dcg)

    def Ndcg(self,gt_items, pred_items):
        IDCG = dcg(gt_items)
        DCG = dcg(pred_items)
        return DCG/IDCG

    def get_metric(self):
        NDCG=[]
        self.model.eval()
        with torch.no_grad():
            for users,pos_items in self.dataloader:
                user_embeddings, pos_item_embeddings,_ = self.model(users=users,
                                                             pos_items=pos_items,
                                                             neg_items=[],
                                                             use_dropout=False)
                all_user_embeddings, all_items_embeddings = self.model.user_embeddings,self.model.item_embeddings
                trained_matrix = torch.matmul(all_user_embeddings,
                                          torch.transpose(all_items_embeddings,0,1))

                _, pred_indices = torch.topk(pos_item_embeddings, self.top_k)

                recommends = torch.take(
                    pos_items, pred_indices).cpu().numpy().tolist()

                _,gt_indices=torch.topk(trained_matrix[user_embeddings],self.top_k)

                ground_truth = torch.take(
                    trained_matrix[user_embeddings],gt_indices).cpu().numpy().tolist()

                NDCG.append(self.Ndcg(gt_items=ground_truth,pred_items=recommends))
        return np.mean(NDCG)
