
import torch
import torch.nn as nn
from utils import Dataloader, EnhancedDict
from models.layers import *
import os

class TRED_GNN20(nn.Module):
    def __init__(self, data:Dataloader, params:EnhancedDict):
        super(TRED_GNN20, self).__init__()
        # 超参数
        self.window_size = params.get("window_size",5)
        self.max_global_window_size = params.get("max_global_window_size",400)
        
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attention_dim = params.attention_dim
        self.data = data
        self.num_relation = data.num_relation
        self.num_entity = data.num_entity
        self.max_history_length = self.data.time_length
        self.time_dim = params.get("time_dim",self.hidden_dim//4)
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': idd}
        act = acts[params.act]
        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(
                TimelineGNNLayer6(self.hidden_dim, self.hidden_dim, self.attention_dim, self.num_relation, act=act,max_history_length=self.max_history_length,time_dim=self.time_dim))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(params.dropout)
        # os.environ['TORCH_USE_CUDA_DSA']='True'
        

    def forward(self, time_stamp:int, subject:torch.Tensor, relation:torch.Tensor):
        """
        
        Parameters:
        -------
        time_stamp:int
            开始追溯时间的起点（不包括该时间点），一般是查询三元组所在的时间点
        subject:Tensor 
            shape=(n,1) 头实体列表
        subject:Tensor 
            shape=(n,1) 尾实体列表
        """
        num_query = subject.shape[0]
        nodes = torch.cat([torch.arange(num_query).unsqueeze(1).cuda(), subject.unsqueeze(1)], 1)
        hidden = torch.zeros(num_query, self.hidden_dim).cuda()
        h0 = torch.zeros((1, num_query, self.hidden_dim)).cuda()

        for i in range(self.n_layer):
            if i==0:
                # 当max_global_window_size==window_size时
                nodes, edges, idx = self.data.get_neighbors_in_period(nodes.data.cpu().numpy(), time_stamp, self.max_global_window_size)
            else:
                nodes, edges, idx = self.data.get_neighbors_in_period(nodes.data.cpu().numpy(), time_stamp, self.window_size)
            hidden = self.gnn_layers[i](subject, relation, hidden, edges, nodes.size(0))

            h0 = torch.zeros(1, nodes.size(0), hidden.size(1),dtype=h0.dtype).cuda().index_copy_(1, idx, h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)

        scores = self.W_final(hidden).squeeze(-1)
        scores_all = torch.zeros((num_query, self.num_entity),dtype=scores.dtype).cuda()
        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores
        return scores_all


