import torch

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

        for epoch in range(epochs):
            avg_cost = 0
            batch_size = len(dataloader)
            for users,pos_items,neg_items in dataloader:
                user_embeddings, pos_item_embeddings, neg_item_embeddings=\
                    model(users,pos_items,neg_items)

                optimizer.zero_grad()
                cost = criterion(user_embeddings,pos_item_embeddings,neg_item_embeddings)
                cost.mean().backward()
                optimizer.step()
                avg_cost += cost.item() / total_batch
            print(f'Epoch: {(epoch + 1):04}, {criterion._get_name()}= {avg_cost:.9f}')




