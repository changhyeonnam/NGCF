import torch
from datetime import datetime
from utils import Download
from utils import MovieLens
from laplacian_mat import Laplacian
from evaluation import Evaluation
from torch.utils.data import DataLoader
from model.NGCF import NGCF
from bpr_loss import BPR_Loss
from train import Train
from parser import args
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')


root_path = 'dataset'
dataset = Download(root=root_path,file_size=args.file_size,download=False)
total_df , train_df, test_df = dataset.split_train_test()

num_user, num_item = total_df['userId'].max()+1, total_df['movieId'].max()+1
train_set = MovieLens(df=train_df,total_df=total_df,train=True,ng_ratio=1)
test_set = MovieLens(df=test_df,total_df=total_df,train=False,ng_ratio=0)

matrix_generator = Laplacian(df=total_df)
eye_matrix,norm_laplacian  = matrix_generator.create_norm_laplacian()

train_loader = DataLoader(train_set,
                          batch_size=args.batch,
                          shuffle=True)

test_loader = DataLoader(test_set,
                         batch_size=100,
                         shuffle=False,
                         drop_last=True
                         )

model = NGCF(norm_laplacian=norm_laplacian,
             eye_matrix= eye_matrix,
             num_user=num_user,
             num_item=num_item,
             embed_size=64,
             device= device,
             node_dropout_ratio=0.1,
             mess_dropout=[0.1,0.1,0.1],
             layer_size=3,
             )

model.to(device)
criterion = BPR_Loss(batch_size=256,decay_ratio=1e-5)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)


if __name__ =='__main__' :
    start_time = datetime.now()
    print('------------train start------------')
    train = Train(device=device,
                  epochs=args.epoch,
                  model=model,
                  train_loader=train_loader,
                  test_loader=test_loader,
                  optim=optimizer,
                  criterion=criterion,
                  top_k=args.top_k,
                  total_df = total_df
                  )
    train.train()
    print('------------train end------------')

    eval = Evaluation(test_dataloader=test_loader,
                      model=model,
                      top_k=args.top_k,
                      device=device)

    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
