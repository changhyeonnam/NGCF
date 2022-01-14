import torch

import gc
class Train():
    def __init__(self,
                 dataloader,
                 model:torch.nn.Module,
                 device:torch.device,
                 criterion,
                 optim:torch.optim,
                 epochs:int) -> object:
        self.epochs =epochs
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optim
        self.dataloader = dataloader

    def train(self):
        epochs = self.epochs
        device = self.device
        model = self.model
        criterion = self.criterion
        optimizer  = self.optimizer
        dataloader = self.dataloader
        device = self.device

        for epoch in range(epochs):
            avg_cost = 0
            batch_size = len(dataloader)
            loss, mf_loss, emb_loss = 0., 0., 0.

            for idx,(users,pos_items,neg_items) in enumerate(dataloader):
                users,pos_items,neg_items = users.to(device),pos_items.to(device),neg_items.to(device)
                user_embeddings, pos_item_embeddings, neg_item_embeddings=\
                    model(users,pos_items,neg_items)

                optimizer.zero_grad()
                batch_loss, batch_mf_loss, batch_emb_loss  = criterion(user_embeddings,pos_item_embeddings,neg_item_embeddings)
                cost = batch_loss
                cost.backward()
                optimizer.step()

                loss += batch_loss
                mf_loss += batch_mf_loss
                emb_loss += batch_emb_loss

            avg_cost = loss/batch_size
            print(f'Epoch: {(epoch + 1):04}, {criterion._get_name()}= {avg_cost:.9f}')




