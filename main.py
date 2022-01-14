import torch
import time
from utils import Download
from utils import MovieLens
from laplacian_mat import Laplacian
from torch.utils.data import DataLoader
from model.NGCF import NGCF
from bpr_loss import BPR_Loss
from train import Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

import gc
gc.collect()
torch.cuda.empty_cache()

root_path = 'dataset'
dataset = Download(root=root_path,file_size='1m',download=False)
total_df , train_df, test_df = dataset.split_train_test()

num_user, num_item = total_df['userId'].max()+1, total_df['movieId'].max()+1
train_set = MovieLens(train_df,total_df,train=True)
test_set = MovieLens(test_df,total_df,train=False)

matrix_generator = Laplacian(df=total_df)
eye_matrix,norm_laplacian  = matrix_generator.create_norm_laplacian()

train_loader = DataLoader(train_set,
                          batch_size=256,
                          shuffle=True)
test_loader = DataLoader(test_set,
                         batch_size=256,
                         shuffle=True)

model = NGCF(norm_laplacian=norm_laplacian,
             eye_matrix= eye_matrix,
             num_user=num_user,
             num_item=num_item,
             embed_size=64,
             device= device,
             node_dropout_ratio=0.1,
             mess_dropout=[0.1,0.1,0.1],
             train=True,
             layer_size=3,
             )

model.to(device)
criterion = BPR_Loss(batch_size=256,decay_ratio=1e-5)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

if __name__ =='__main__' :
    start = time.time()

    train = Train(device=device,
                  epochs=10,
                  model=model,
                  dataloader=train_loader,
                  optim=optimizer,
                  criterion=criterion
                  )
    train.train()
    end = time.time()
    print(f'training time:{end-start:.5f}')
