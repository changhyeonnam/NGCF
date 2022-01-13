import torch
from utils import Download
from utils import MovieLens
from laplacian_mat import Laplacian


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

root_path = 'dataset'
dataset = Download(root=root_path,file_size='100k',download=False)
total_df , train_df, test_df = dataset.split_train_test()

train_set = MovieLens(train_df,total_df,train=True)
test_set = MovieLens(test_df,total_df,train=False)

matrix_generator = Laplacian(df=total_df)
norm_laplacian  = matrix_generator.create_norm_laplacian()
print(norm_laplacian)