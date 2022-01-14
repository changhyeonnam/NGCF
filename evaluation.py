import torch
import torch.nn as nn
import numpy as np
import math
class Evaluation():
    def __init__(self,
                 test_dataloader,
                 model,
                 device,
                 top_k:int=5,
                 ):
        self.dataloader = test_dataloader
        self.model = model
        self.top_k = top_k
        self.device = device
    def dcg(self,gt_items):
        dcg=[]
        for idx,item in enumerate(gt_items):
            dcg.append(pow(2,item)/math.log(idx+2,2))
        return np.sum(dcg)

    def Ndcg(self,gt_items, pred_items):
        IDCG = self.dcg(gt_items)
        DCG = self.dcg(pred_items)
        output = DCG/IDCG
        return output

    def get_metric(self):
        NDCG=[]
        device = self.device
        self.model.eval()
        with torch.no_grad():
            for users,pos_items in self.dataloader:
                users,pos_items = users.to(device),pos_items.to(device)

                users=torch.flatten(users,start_dim=0)
                pos_items=torch.flatten(pos_items,start_dim=0)

                user_embeddings, pos_item_embeddings,_ = self.model(users=users,
                                                             pos_items=pos_items,
                                                             neg_items=[],
                                                             use_dropout=False)


                all_user_embeddings, all_items_embeddings = self.model.user_embeddings,self.model.item_embeddings

                trained_matrix = torch.matmul(all_user_embeddings,
                                          torch.transpose(all_items_embeddings,0,1))

                pred_matrix = torch.matmul(user_embeddings,torch.transpose(pos_item_embeddings,0,1))


                _, pred_indices = torch.topk(pred_matrix[0], self.top_k)
                recommends = torch.take(
                    pred_matrix[0], pred_indices).cpu().numpy().tolist()

                _,gt_indices=torch.topk(trained_matrix[users[0]],self.top_k)
                
                ground_truth = torch.take(
                    trained_matrix[users[0]],gt_indices).cpu().numpy().tolist()
                
                NDCG.append(self.Ndcg(gt_items=ground_truth,pred_items=recommends))
                
        return np.mean(NDCG)
