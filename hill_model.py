import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import random
from math import sqrt

class HillModel(nn.Module):
    def __init__(self, n_feature, train_gsize):
        super(HillModel, self).__init__()
        self.class_features = None
        self.register_buffer('mlayer', torch.eye(n_feature))
        alphap, alphan = torch.FloatTensor(train_gsize), torch.FloatTensor(train_gsize)
        alphap[0] = 1
        for i in range(1, train_gsize):
            alphap[i] = alphap[i - 1] + 1.0 / (1 + i)
        for i in range(train_gsize):
            alphan[i] = (train_gsize - i) * 1.0 / (train_gsize - 1)
        self.alphap, self.alphan = alphap, alphan
        self.f_scores = []
        self.fscores_sort = [] 
        self.fscores_sort_idx = []
    
    def setClassFeatures(self, mean_features):
        self.mean_features = mean_features    # [class_num, n_fea]

    def forward(self, probe_fea, probe_label):
        '''
        probe_fea : [1, n_fea]
        probe_label : [1]
        '''
        batch_size = probe_fea.shape[0]
        class_num = self.mean_features.shape[0]
        n_fea = probe_fea.shape[1]
        gallery_features = self.mean_features
        f_scores = torch.FloatTensor()
        for i in range(batch_size):
            diff_xp_xg = -gallery_features + probe_fea[i] # [class_num, n_fea]
            diff_mm = torch.mm(diff_xp_xg, self.mlayer) # [class_num, n_feature]
            diff_sum = torch.sum(torch.mul(diff_mm, diff_xp_xg), dim=1)  # [class_num]
            diff_sum = -diff_sum.unsqueeze(0)
            if i == 0:
                f_scores = diff_sum
            else:
                f_scores = torch.cat((f_scores, diff_sum), dim = 0)
        return f_scores

    def get_rank_list(self, probe_fea):
        
        gallery_features = self.mean_features
        diff_xp_xg = -gallery_features + probe_fea # [class_num, n_fea]
        diff_mm = torch.mm(diff_xp_xg, self.mlayer) # [class_num, n_feature]
        diff_sum = torch.sum(torch.mul(diff_mm, diff_xp_xg), dim=1)  # [class_num]
        self.f_scores = -diff_sum
        self.fscores_sort, self.fscores_sort_idx = torch.sort(self.f_scores, dim=0, descending=True) # sort from small to large
        
        return self.f_scores, self.fscores_sort_idx

    def humanSelectPositive(self, probe_fea, g_index):
        '''
        gallery_labels : [gsize,]
        probe_label : [1,]
        '''
        # update the M matrix for positive sample
        gallery_features = self.mean_features
        g_rank = torch.nonzero(self.fscores_sort_idx == g_index, as_tuple=False)[0][0]
        m_violator_idx = self.fscores_sort_idx[0]

        #update the M matrix
        m_b = self.mlayer
        eta = 0.05
        f_n = self.f_scores[g_index]
        L_n = self.alphap[g_rank]
        f_t = f_n / (1 - eta * L_n * f_n)
        z_t = probe_fea - gallery_features[g_index] # [1, n_fea]
        z_t_T = z_t.permute(1, 0)   # [n_fea, 1]
        delta_up = eta * L_n * torch.mm(torch.mm(m_b, torch.mm(z_t_T, z_t)), m_b)
        delta_down = 1.0 + eta * L_n * torch.mm(torch.mm(z_t, m_b), z_t_T)
        self.mlayer = m_b - delta_up / delta_down

    def humanSelectNegative(self, probe_fea, g_index_n):
        # update the M matrix for negative sample
        gallery_features = self.mean_features
        g_rank_n = torch.nonzero(self.fscores_sort_idx == g_index_n, as_tuple=False)[0][0]

        close_to_idx = 20
        m_violator_idx_n = self.fscores_sort_idx[close_to_idx]
        m_b = self.mlayer
        eta = 0.5
        b_t = -1
        L_n = self.alphan[g_rank_n]
        f_v = self.f_scores[m_violator_idx_n]
        f_n = self.f_scores[g_index_n]
        itm = eta * L_n * (f_v + b_t) * f_n - 1
        f_t = (itm + sqrt(itm * itm + 4 * eta * L_n * f_n * f_n)) / (2 * eta * L_n * f_n)
        z_t = probe_fea - gallery_features[g_index_n] # [1, 512]
        z_t_T = z_t.permute(1, 0)
        itm2 = eta * L_n * (b_t - f_t + f_v)
        delta_up = itm2 * torch.mm(torch.mm(m_b, torch.mm(z_t_T, z_t)), m_b)
        delta_down = 1.0 + itm2 * torch.mm(torch.mm(z_t, m_b), z_t_T)
        self.mlayer = nn.Parameter(m_b - delta_up / delta_down)

if __name__ == '__main__':
    model = HillModel()