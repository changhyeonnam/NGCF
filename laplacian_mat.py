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

    def create_laplace_matrix(self):
