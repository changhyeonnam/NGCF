import torch
from utils import Download
from utils import MovieLens



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

root_path = 'dataset'
dataset = Download(root=root_path,file_size='1m',download=True)
total_df , train_df, test_df = dataset.split_train_test()

train_set = MovieLens(train_df,total_df,train=True)
test_set = MovieLens(test_df,total_df,train=False)

