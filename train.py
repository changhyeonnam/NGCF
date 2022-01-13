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
            batch_size = len(dataloader)
            loss, mf_loss, emb_loss = 0., 0., 0.

            for users,pos_items,neg_items in dataloader:
                user_embeddings, pos_item_embeddings, neg_item_embeddings=\
                    model(users,pos_items,neg_items)

                batch_loss, batch_mf_loss, batch_emb_loss = criterion(user_embeddings,
                                                                      pos_item_embeddings,
                                                                      neg_item_embeddings)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss
                mf_loss += batch_mf_loss
                emb_loss += batch_emb_loss

            if (epoch + 1) % 10 != 0:
                if args.verbose > 0 and epoch % args.verbose == 0:
                    perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                        epoch, time() - t1, loss, mf_loss, emb_loss)
                    print(perf_str)
                continue


