import torch
import torch.nn as nn
import scipy.sparse as sp
class NGCF(nn.Module):
    def  __init__(self,
                  norm_laplacian:sp.coo_matrix,
                  num_user:int,
                  num_item:int,
                  embed_size:int,
                  node_dropout_ratio:float,
                  mess_dropout:float,
                  layer_size:int,
                  decay_ratio:float,
                  train:bool
                  ):
        super(NGCF, self).__init__()

        self.num_users = num_user
        self.num_items = num_item
        self.embed_size = embed_size
        self.node_dropout_ratio = node_dropout_ratio
        self.mess_dropout = mess_dropout
        self.layer_size = layer_size
        self.decay_ratio = decay_ratio
        self.train = train
        self.embedding_user = nn.Embedding(self.num_users,self.embed_size)
        self.embedding_item = nn.Embedding(self.num_items,self.embed_size)

        layer_list=[]
        for index in range(self.layer_size):
            layer_list.append(nn.Linear(64))

        self.layer1 = nn.Sequential(*layer_list)
        self.layer2 = nn.Sequential(*layer_list)
        self._init_weight()

        self.norm_laplacian = self._covert_mat2tensor(norm_laplacian)

    def _init_weight(self):
        nn.init.xavier_uniform(self.embedding_user.weight)
        nn.init.xavier_uniform(self.embedding_item.weight)

        for layer in self.layer1:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform(layer.weight)
                nn.init.xavier_uniform(layer.bias)

        for layer in self.layer2:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform(layer.weight)
                nn.init.xavier_uniform(layer.bias)
    def _node_dropout(self,rate):
        radom_tensor = (1-rate)+torch.range(self.norm_laplacian.shape)
        dropout_mask = torch.floor(radom_tensor).type(bool)
        indices = self.norm_laplacian._indices()
        values = self.norm_laplacian._values()
        indices = i[:,dropout_mask]
        values = v[dropout_mask]
        output = torch.sparse.FloatTensor(indices,values,self.norm_laplacian.shape)
        return output * (1./(1-rate))

    def _sparse_eye(self,mat):
        indices = mat._indices()
        values = torch.tensor(1.0)


    def _covert_mat2tensor(self,mat):
        mat = mat.tocoo()
        indices = torch.LongTensor([mat.row,mat.col])
        values = torch.from_numpy(mat.data).float()
        return torch.sparse.FloatTensor(indices,values,coo.shape)

    def forward(self,users, pos_items, neg_items):

        norm_laplacian = self._node_dropout(self.node_dropout_ratio if self.train
                                            else self.norm_laplacian)
        eye_mat=torch.eye(self.num_users+self.num_items)
        embedding_zero = torch.cat((self.embedding_user.weight,
                                    self.embedding_item.weight),dim=0)
        all_embedding=[]
        all_embedding.append(embedding_zero)

        for index in range(self.layer_size):





