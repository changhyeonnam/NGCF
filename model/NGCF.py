import torch
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
import numpy as np

class NGCF(nn.Module):
    def  __init__(self,
                  norm_laplacian: sp.coo_matrix,
                  eye_matrix:sp.coo_matrix,
                  num_user: int,
                  num_item: int,
                  embed_size: int,
                  node_dropout_ratio: float,
                  layer_size: int,
                  device,
                  mess_dropout= [0.1, 0.1, 0.1],
                  ) -> object:
        super(NGCF, self).__init__()

        self.num_users = num_user
        self.num_items = num_item
        self.embed_size = embed_size
        self.node_dropout_ratio = node_dropout_ratio
        self.mess_dropout = mess_dropout
        self.layer_size = layer_size
        self.embedding_user = nn.Embedding(self.num_users,self.embed_size)
        self.embedding_item = nn.Embedding(self.num_items,self.embed_size)

        layer_list=[]
        for index in range(self.layer_size):
            layer_list.append(nn.Linear(64,64))

        self.layer1 = nn.Sequential(*layer_list)
        self.layer2 = nn.Sequential(*layer_list)
        self._init_weight()

        self.user_embeddings = None
        self.item_embeddings = None
        # convert coordinate representation to sparse matrix
        self.norm_laplacian = self._covert_mat2tensor(norm_laplacian).to(device)
        self.eye_matrix = self._covert_mat2tensor(eye_matrix).to(device)

    def _init_weight(self):
        nn.init.xavier_uniform(self.embedding_user.weight)
        nn.init.xavier_uniform(self.embedding_item.weight)

        for layer in self.layer1:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform(layer.weight.data)
                layer.bias.data.fill_(0.01)

        for layer in self.layer2:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform(layer.weight.data)
                layer.bias.data.fill_(0.01)

    def _node_dropout(self,mat):
        node_mask = nn.Dropout(self.node_dropout_ratio)\
            (torch.tensor(np.ones(mat._nnz()))).type(torch.bool)
        indices = mat._indices()
        values = mat._values()
        indices = indices[:, node_mask]
        values = values[node_mask]
        out = torch.sparse.FloatTensor(indices, values, mat.shape)
        return out

    def _covert_mat2tensor(self,mat):
        indices = torch.LongTensor([mat.row,mat.col])
        values = torch.from_numpy(mat.data).float()
        return torch.sparse.FloatTensor(indices,values,mat.shape)


    def forward(self,users, pos_items, neg_items,use_dropout):

        # node dropout
        if use_dropout:
            norm_laplacian = self._node_dropout(self.norm_laplacian)
        else:
            norm_laplacian = self.norm_laplacian

        # print(f'norm_laplacian:{norm_laplacian.shape}')


        # identity matrix
        norm_laplacian_add_eye = norm_laplacian+self.eye_matrix

        # make coordinate format
        # norm_laplacian_add_eye = norm_laplacian_add_eye.to_sparse()

        # print(f'norm_laplacian_add_eye:{norm_laplacian_add_eye.shape}')

        prev_embedding= torch.cat((self.embedding_user.weight,
                                    self.embedding_item.weight),dim=0)

        # print(f'0-th_prev_embedding:{prev_embedding.shape}')

        all_embedding=[prev_embedding]

        for index,(l1,l2) in enumerate(zip(self.layer1,self.layer2)):

            # first_term = (laplacian + identity) * previous embedding
            first_term= torch.sparse.mm(norm_laplacian_add_eye,prev_embedding)

            # first_term = first_embedding * i-th layer of W_1
            first_term = torch.matmul(first_term,l1.weight)+l1.bias
            # print(f'{index+1}-th_first_term:{first_term.shape}')

            # second_term = laplacian * elementwise of previous embedding
            second_term = prev_embedding * prev_embedding
            second_term = torch.sparse.mm(norm_laplacian,second_term)

            # second_term = second_embedding * i-th layer of W2
            second_term = torch.matmul(second_term,l2.weight)+l2.bias
            # print(f'{index+1}-th_second_term:{second_term.shape}')

            # prev_embedding = LeakyReLU(first_term + second_term)
            prev_embedding = nn.LeakyReLU(negative_slope=0.2)(first_term+second_term)

            # message dropout
            prev_embedding = nn.Dropout(self.mess_dropout[index])(prev_embedding)

            # L2 normalize
            prev_embedding = F.normalize(prev_embedding,p=2,dim=1)
            # print(f'{index+1}-th_prev_embedding:{prev_embedding.shape}')

            all_embedding+=[prev_embedding]

        all_embedding = torch.cat(all_embedding,1)
        # print(f'all_embedding:{all_embedding.shape}')

        self.user_embeddings = all_embedding[:self.num_users,:]
        # print(f'user_embeddings:{user_embeddings.shape}')

        self.item_embeddings = all_embedding[self.num_users:,:]
        # print(f'item_embeddings:{item_embeddings.shape}')

        users_embed=self.user_embeddings[users,:]
        pos_item_embeddings = self.item_embeddings[pos_items, :]
        neg_item_embeddings = self.item_embeddings[neg_items, :]

        # print(f'users_embed:{users_embed.shape}')
        # print(f'pos_item_embeddings:{pos_item_embeddings.shape}')
        # print(f'neg_item_embeddings:{neg_item_embeddings.shape}')

        return users_embed,pos_item_embeddings,neg_item_embeddings








