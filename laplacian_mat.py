import scipy.sparse as sp
import pandas as pd
import numpy as  np

class Laplacian():
    def __init__(self,
                 df:pd.DataFrame,):
        self.df = df
        self.n_users = df["userId"].max()+1
        self.n_items = df["movieId"].max()+1
        self.R = self.create_interaction_matrix()
        self.adj_mat = self.create_adjacent_matrix()

    def create_interaction_matrix(self):
        df = self.df
        R_mat = sp.dok_matrix((self.n_users,self.n_items),dtype=np.float32)
        user_item_set = set(zip(df['userId'], df['movieId']))
        for u, i in user_item_set:
            R_mat[u,i]=1
        return R_mat

    def create_adjacent_matrix(self):
        adj_mat = sp.dok_matrix((self.n_users+self.n_items,self.n_users+self.n_items),dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()
        adj_mat[:self.n_users,self.n_users:]=R
        adj_mat[self.n_users:,:self.n_users]=R.T
        adj_mat = adj_mat.todok()
        return adj_mat

    def create_norm_laplacian(self):
        adj_mat = self.adj_mat
        rowsum = np.array(adj_mat.sum(axis=1)) # axis =1 -> sum the rows, axis=0 -> sum the columns
        deg_inverse = np.power(rowsum,-0.5).flatten()
        deg_inverse[np.isinf(deg_inverse)] = 0. # check if sqrt(-1) exits
        deg_inverse_mat = sp.diags(deg_inverse)
        norm_laplacian = deg_inverse_mat.dot(adj_mat).dot(deg_inverse_mat)

        eye_matrix = sp.eye(self.n_users+self.n_items)
        eye_matrix = eye_matrix.tocoo()
        # tocsr contains only row index for non-zero value
        # tocoo contains row,column for non-zero value
        return eye_matrix,norm_laplacian.tocoo()




