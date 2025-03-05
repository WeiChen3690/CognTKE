
import torch
import torch.nn as nn
from torch_scatter import scatter
import math
# from torch_geometric.nn.aggr import SetTransformerAggregation

def idd(x):
    return x

class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=idd):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        

        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, q_sub, q_rel, hidden, edges, n_node, old_nodes_new_idx):
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]

        hs = hidden[sub]
        hr = self.rela_embed(rel)

        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]

        message = hs + hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new


class TGNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=idd, window_size=5):
        super(TGNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        self.time_embed = nn.Embedding(window_size+1, in_dim//4)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.fuse_mlp = nn.Sequential(nn.Linear(in_dim//4*5, in_dim),nn.LeakyReLU(),nn.Linear(in_dim, in_dim),nn.LeakyReLU())
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, q_sub, q_rel, hidden, edges, n_node):
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]
        output, reverse_indexes = edges[:,[2,6]].unique(dim=0,return_inverse=True, sorted=True)
        temp_rel_emb = self.rela_embed(output[:,0])
        temp_time_emb = self.time_embed(output[:,1])
        temp_comp_raw = torch.concatenate([temp_rel_emb,temp_time_emb],dim=1)
        temp_comp = self.fuse_mlp(temp_comp_raw)+temp_rel_emb
        hr = temp_comp[reverse_indexes]
        hs = hidden[sub]
        # hr = self.rela_embed(rel)

        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]

        message = hs + hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new
    
class GNNLayer2(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=idd):
        super(GNNLayer2, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        

        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, q_sub, q_rel, hidden, edges, n_node):
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]

        hs = hidden[sub]
        hr = self.rela_embed(rel)

        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]

        message = hs + hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new
    

class TimelineGNNLayer5(torch.nn.Module):
    """
    采用基于正弦余弦函数计算的相对时间编码，实际上应该考虑相对时间和绝对时间混合。毕竟有些事情的发生是限定了发生的时间的。（工作日）
    进一步改进聚合方式,加上QRFGU, 使用sigmoid, 并在聚合后根据聚合的数量做了平均
    """
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=idd, max_history_length=1000):
        super(TimelineGNNLayer5, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        self.time_embed = TimeEncoding(hidden_dim=in_dim//4,max_length=max_history_length)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.fuse_mlp = nn.Sequential(nn.Linear(in_dim//4*5, in_dim),nn.LeakyReLU(),nn.Linear(in_dim, in_dim),nn.LeakyReLU())
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1,bias=False)
        self.gate = GateUnit(in_dim, in_dim)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, q_sub, q_rel, hidden, edges, n_node):
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]
        output, reverse_indexes = edges[:,[2,6]].unique(dim=0,return_inverse=True, sorted=True)
        temp_rel_emb = self.rela_embed(output[:,0])
        temp_time_emb = self.time_embed(output[:,1])
        temp_comp_raw = torch.concatenate([temp_rel_emb,temp_time_emb],dim=1)
        temp_comp = self.fuse_mlp(temp_comp_raw)+temp_rel_emb
        hr = temp_comp[reverse_indexes]
        hs = hidden[sub]
        # hr = self.rela_embed(rel)

        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]
        # 消息函数也需要更换
        message = self.gate(hr, h_qr, hs)
        # 注意力应该可以换成1层自注意力+1层其他注意力
        alpha = self.w_alpha(self.leakyrelu(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr)))
        sigmoid_attention = torch.sigmoid(alpha)
        up_message = sigmoid_attention * message

        message_agg = scatter(up_message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        ones = torch.ones(size=(up_message.shape[0],1),dtype=torch.float32,device=message_agg.device)
        degrees = scatter(ones, index=obj, dim=0, dim_size=n_node, reduce='sum')
        message_agg = message_agg/torch.sqrt(degrees+1e-4)
        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new
    

class TimelineGNNLayer6(torch.nn.Module):
    """
    设置了自由的时间维度，使用gate、sigmoid，只使用相对时间编码
    """
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=idd, max_history_length=1000,time_dim=-1):
        super(TimelineGNNLayer6, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        if time_dim<=0:
            self.time_dim = in_dim//4
        else:
            self.time_dim = time_dim
        self.time_embed = TimeEncoding(hidden_dim=self.time_dim,max_length=max_history_length)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.fuse_mlp = nn.Sequential(nn.Linear(in_dim+self.time_dim, in_dim),nn.LeakyReLU(),nn.Linear(in_dim, in_dim),nn.LeakyReLU())
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1,bias=False)
        self.gate = GateUnit(in_dim, in_dim)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, q_sub, q_rel, hidden, edges, n_node):
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]
        output, reverse_indexes = edges[:,[2,6]].unique(dim=0,return_inverse=True, sorted=True)
        temp_rel_emb = self.rela_embed(output[:,0])
        temp_time_emb = self.time_embed(output[:,1])
        temp_comp_raw = torch.concatenate([temp_rel_emb,temp_time_emb],dim=1)
        temp_comp = self.fuse_mlp(temp_comp_raw)+temp_rel_emb
        hr = temp_comp[reverse_indexes]
        hs = hidden[sub]


        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]

        message = self.gate(hr, h_qr, hs)
        # 注意力应该可以换成1层自注意力+1层其他注意力
        alpha = self.w_alpha(self.leakyrelu(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr)))
        sigmoid_attention = torch.sigmoid(alpha)
        up_message = sigmoid_attention * message

        message_agg = scatter(up_message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        ones = torch.ones(size=(up_message.shape[0],1),dtype=torch.float32,device=message_agg.device)
        degrees = scatter(ones, index=obj, dim=0, dim_size=n_node, reduce='sum')
        message_agg = message_agg/torch.sqrt(degrees+1e-4)
        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new
    
class TimelineGNNLayer12(torch.nn.Module):
    """
    设置了自由的时间维度，使用gate、sigmoid，只使用相对时间编码
    """
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=idd, max_history_length=1000,time_dim=-1):
        super(TimelineGNNLayer12, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        if time_dim<=0:
            self.time_dim = in_dim//4
        else:
            self.time_dim = time_dim
        self.time_embed = TimeEncoding(hidden_dim=self.time_dim,max_length=max_history_length)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.fuse_mlp = nn.Sequential(nn.Linear(in_dim+self.time_dim, in_dim),nn.LeakyReLU(),nn.Linear(in_dim, in_dim),nn.LeakyReLU())
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1,bias=False)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, q_sub, q_rel, hidden, edges, n_node):
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]
        output, reverse_indexes = edges[:,[2,6]].unique(dim=0,return_inverse=True, sorted=True)
        temp_rel_emb = self.rela_embed(output[:,0])
        temp_time_emb = self.time_embed(output[:,1])
        temp_comp_raw = torch.concatenate([temp_rel_emb,temp_time_emb],dim=1)
        temp_comp = self.fuse_mlp(temp_comp_raw)+temp_rel_emb
        hr = temp_comp[reverse_indexes]
        hs = hidden[sub]


        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]

        message = hr+ hs
        # 注意力应该可以换成1层自注意力+1层其他注意力
        alpha = self.w_alpha(self.leakyrelu(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr)))
        sigmoid_attention = torch.sigmoid(alpha)
        up_message = sigmoid_attention * message

        message_agg = scatter(up_message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        ones = torch.ones(size=(up_message.shape[0],1),dtype=torch.float32,device=message_agg.device)
        degrees = scatter(ones, index=obj, dim=0, dim_size=n_node, reduce='sum')
        message_agg = message_agg/torch.sqrt(degrees+1e-4)
        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new
    
class GNNLayer6(torch.nn.Module):
    """
    设置了自由的时间维度，使用gate、sigmoid，只使用相对时间编码
    """
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=idd, max_history_length=1000,time_dim=-1):
        super(GNNLayer6, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        if time_dim<=0:
            self.time_dim = in_dim//4
        else:
            self.time_dim = time_dim
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.fuse_mlp = nn.Sequential(nn.Linear(in_dim+self.time_dim, in_dim),nn.LeakyReLU(),nn.Linear(in_dim, in_dim),nn.LeakyReLU())
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1,bias=False)
        self.gate = GateUnit(in_dim, in_dim)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, q_sub, q_rel, hidden, edges, n_node):
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]
        output, reverse_indexes = edges[:,[2,6]].unique(dim=0,return_inverse=True, sorted=True)
        temp_rel_emb = self.rela_embed(output[:,0])
        hr = temp_rel_emb[reverse_indexes]
        hs = hidden[sub]


        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]

        message = self.gate(hr, h_qr, hs)
        # 注意力应该可以换成1层自注意力+1层其他注意力
        alpha = self.w_alpha(self.leakyrelu(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr)))
        sigmoid_attention = torch.sigmoid(alpha)
        up_message = sigmoid_attention * message

        message_agg = scatter(up_message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        ones = torch.ones(size=(up_message.shape[0],1),dtype=torch.float32,device=message_agg.device)
        degrees = scatter(ones, index=obj, dim=0, dim_size=n_node, reduce='sum')
        message_agg = message_agg/torch.sqrt(degrees+1e-4)
        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new
    
    
class TimelineGNNLayer9(torch.nn.Module):

    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=idd, max_history_length=1000,time_dim=-1):
        super(TimelineGNNLayer9, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        if time_dim<=0:
            self.time_dim = in_dim//4
        else:
            self.time_dim = time_dim
        self.time_embed = TimeEncoding(hidden_dim=self.time_dim,max_length=max_history_length)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.fuse_mlp = nn.Sequential(nn.Linear(in_dim+self.time_dim, in_dim),nn.LeakyReLU(),nn.Linear(in_dim, in_dim),nn.LeakyReLU())
        self.Wqr_attn = nn.Linear(3*in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1,bias=False)
        self.gate = GateUnitExpand(factor_size= 3*in_dim, hidden_size= in_dim)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, q_sub:torch.Tensor, q_rel:torch.Tensor, hidden:torch.Tensor, edges:torch.Tensor, n_node:int, edge_head_rc_repr, edge_tail_rc_repr, query_head_rc_repr):
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]
        output, reverse_indexes = edges[:,[2,6]].unique(dim=0,return_inverse=True, sorted=True)
        temp_rel_emb = self.rela_embed(output[:,0])
        temp_time_emb = self.time_embed(output[:,1])
        temp_comp_raw = torch.concatenate([temp_rel_emb,temp_time_emb],dim=1)
        temp_comp = self.fuse_mlp(temp_comp_raw)+temp_rel_emb
        hr = temp_comp[reverse_indexes]
        hs = hidden[sub]
        # hr = self.rela_embed(rel)

        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]
        affect_factor = torch.concat([h_qr, edge_head_rc_repr, edge_tail_rc_repr],dim=1)
        # 消息函数也需要更换
        message = self.gate(hr, affect_factor, hs)
        # 注意力应该可以换成1层自注意力+1层其他注意力
        alpha = self.w_alpha(self.leakyrelu(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(affect_factor)))
        sigmoid_attention = torch.sigmoid(alpha)
        up_message = sigmoid_attention * message

        message_agg = scatter(up_message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        ones = torch.ones(size=(up_message.shape[0],1),dtype=torch.float32,device=message_agg.device)
        degrees = scatter(ones, index=obj, dim=0, dim_size=n_node, reduce='sum')
        message_agg = message_agg/torch.sqrt(degrees+1e-4)
        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new
    


    

class InitRelationGNNLayer(torch.nn.Module):
    """
    编码关系邻域表示的GNN层

    """

    def __init__(self, hidden_dim,time_dim, n_rel, max_history_length):
        super(InitRelationGNNLayer, self).__init__()
        self.in_dim = hidden_dim
        self.out_dim = hidden_dim
        self.relu = nn.ReLU()

        self.W_h = nn.Linear(2*hidden_dim, hidden_dim)
        self.W_a = nn.Linear(hidden_dim,1,bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

        
        if time_dim<=0:
            self.time_dim = hidden_dim//4
        else:
            self.time_dim = time_dim
        self.time_embed = TimeEncoding(hidden_dim=self.time_dim,max_length=max_history_length)
        self.fuse_mlp = nn.Sequential(nn.Linear(hidden_dim+self.time_dim, hidden_dim),nn.LeakyReLU(),nn.Linear(hidden_dim, hidden_dim),nn.LeakyReLU())
        self.rela_embed = nn.Embedding(2 * n_rel + 1, hidden_dim)


    def forward(self, edges, hiddens):
        # 因为头尾实体集合包含的实体类型相同，生成新编号时也经过了排序，所以二者应该是同一个编号系
        # 这个方法不会改变hiddens的长度
        # sub(0),rel(1),obj(2),sub2(3),obj2(4),time(5)
        objs = edges[:,4]
        # 选取互斥的关系-时间组合
        output, reverse_indexes = edges[:,[1,5]].unique(dim=0,return_inverse=True, sorted=True)
        temp_rel_emb = self.rela_embed(output[:,0])
        temp_time_emb = self.time_embed(output[:,1])
        temp_comp_raw = torch.concatenate([temp_rel_emb,temp_time_emb],dim=1)
        temp_comp = self.fuse_mlp(temp_comp_raw)+temp_rel_emb
        hr = temp_comp[reverse_indexes]
        hs = hiddens[edges[:,3]]

        message = self.W_h(torch.cat([hs,hr],dim=-1))
        attention = self.W_a(hr)
        attention_exp = torch.exp(attention)
        #  +1e-6防止分母为0
        eps = 1e-6
        ent_agg_repr = scatter(message*attention_exp, objs,dim=0, dim_size=hiddens.shape[0])+eps
        
        attention_sum = scatter(attention,objs,dim=0, dim_size=hiddens.shape[0])+eps
        ent_out_repr =  self.norm(ent_agg_repr/attention_sum)
        return ent_out_repr
    
class InitRelationGNNLayer2(torch.nn.Module):
    """
    编码关系邻域表示的GNN层

    """

    def __init__(self, hidden_dim,time_dim, n_rel, max_history_length):
        super(InitRelationGNNLayer2, self).__init__()
        self.in_dim = hidden_dim
        self.out_dim = hidden_dim
        self.relu = nn.ReLU()

        self.W_h = nn.Linear(2*hidden_dim, hidden_dim)
        self.W_a = nn.Linear(hidden_dim,1,bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(hidden_dim)

        
        if time_dim<=0:
            self.time_dim = hidden_dim//4
        else:
            self.time_dim = time_dim
        self.time_embed = TimeEncoding(hidden_dim=self.time_dim,max_length=max_history_length)
        self.fuse_mlp = nn.Sequential(nn.Linear(hidden_dim+self.time_dim, hidden_dim),nn.LeakyReLU(),nn.Linear(hidden_dim, hidden_dim),nn.LeakyReLU())
        self.rela_embed = nn.Embedding(2 * n_rel + 1, hidden_dim)


    def forward(self, edges, hiddens):
        # 因为头尾实体集合包含的实体类型相同，生成新编号时也经过了排序，所以二者应该是同一个编号系
        # 这个方法不会改变hiddens的长度
        # sub(0),rel(1),obj(2),sub2(3),obj2(4),time(5)
        objs = edges[:,4]
        # 选取互斥的关系-时间组合
        output, reverse_indexes = edges[:,[1,5]].unique(dim=0,return_inverse=True, sorted=True)
        temp_rel_emb = self.rela_embed(output[:,0])
        temp_time_emb = self.time_embed(output[:,1])
        temp_comp_raw = torch.concatenate([temp_rel_emb,temp_time_emb],dim=1)
        temp_comp = self.fuse_mlp(temp_comp_raw)+temp_rel_emb
        hr = temp_comp[reverse_indexes]
        hs = hiddens[edges[:,3]]

        message = self.W_h(torch.cat([hs,hr],dim=-1))
        attention = self.W_a(message)
        attention_sigmoid = self.sigmoid(attention)
        ent_agg_repr = scatter(message*attention_sigmoid, objs,dim=0, dim_size=hiddens.shape[0])

        ent_out_repr =  self.norm(ent_agg_repr)
        return ent_out_repr


    

    

class TimeEncoding(torch.nn.Module):
    def __init__(self, hidden_dim, max_length=1000):
        super().__init__()
        self.d_model = hidden_dim
        self.max_length = max_length
        
        # 创建嵌入矩阵
        self.embedding = torch.nn.Embedding(max_length, hidden_dim)
        
        # 计算正弦和余弦函数的值
        pos = torch.arange(0, max_length).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_length, hidden_dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        
        # 在嵌入矩阵中初始化编码
        self.embedding.weight.data = pe
        self.embedding.weight.requires_grad = False
        
    def forward(self, x):
        # 将时间值映射为嵌入表示
        x = self.embedding(x)
        return x
    

class GateUnit(nn.Module):
    """
    控制新的信息能够添加到旧有表示
    # todo 替换为使用fc层的模块
    """

    def __init__(self, factor_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_W = nn.Sequential(nn.Linear(self.hidden_size * 2+factor_size, self.hidden_size * 2),
                                  nn.Sigmoid())
        self.hidden_trans = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Tanh()
        )

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, message: torch.Tensor, factor: torch.Tensor, hidden_state: torch.Tensor)->torch.Tensor:
        """
        通过类似GRU的门控机制更新实体表示

        :param message: message[batch_size,input_size]
        :param query_r: query_r[batch_size,input_size]
        :param hidden_state: if it is none,it will be allocated a zero tensor hidden state
        :return:
        """
        factors = torch.cat([message, factor, hidden_state], dim=1)
        # 计算门, 计算门时考虑到查询的关系
        update_value, reset_value = self.gate_W(factors).chunk(2, dim=1)
        # 计算候选隐藏表示
        hidden_candidate = self.hidden_trans(torch.cat([message, reset_value * hidden_state], dim=1))
        hidden_state = (1 - update_value) * hidden_state + update_value * hidden_candidate
        return hidden_state

class GateUnitSimple(nn.Module):
    """
    控制新的信息能够添加到旧有表示
    
    简化版，似乎也有效果
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate = nn.Sequential(nn.Linear(self.hidden_size * 3, self.hidden_size ),
                                  nn.Sigmoid())
        self.hidden_trans = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Tanh()
        )

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, message: torch.Tensor, query_r: torch.Tensor, hidden_state: torch.Tensor)->torch.Tensor:
        """
        通过类似GRU的门控机制更新实体表示

        :param message: message[batch_size,input_size]
        :param query_r: query_r[batch_size,input_size]
        :param hidden_state: if it is none,it will be allocated a zero tensor hidden state
        :return:
        """
        # 计算门, 计算门时考虑到查询的关系
        reset_value = self.gate(torch.cat([message, query_r, hidden_state], dim=1))
        # 计算候选隐藏表示
        hidden_candidate = self.hidden_trans(torch.cat([message, reset_value * hidden_state], dim=1))
        return hidden_candidate
    

class GateUnitExpand(nn.Module):
    """
    拓展的gate单元，支持自由的因素大小。
    """

    def __init__(self, factor_size, hidden_size, div=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_W = nn.Sequential(nn.Linear(self.hidden_size * 2+factor_size, self.hidden_size * 2),
                                  nn.Sigmoid())
        self.hidden_trans = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Tanh()
        )
        self.div = div
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, message: torch.Tensor, factor: torch.Tensor, hidden_state: torch.Tensor)->torch.Tensor:
        """
        通过类似GRU的门控机制更新实体表示

        :param message: message[batch_size,input_size]
        :param query_r: query_r[batch_size,input_size]
        :param hidden_state: if it is none,it will be allocated a zero tensor hidden state
        :return:
        """
        factors = torch.cat([message, factor/self.div, hidden_state], dim=1)
        # 计算门, 计算门时考虑到查询的关系
        update_value, reset_value = self.gate_W(factors).chunk(2, dim=1)
        # 计算候选隐藏表示
        hidden_candidate = self.hidden_trans(torch.cat([message, reset_value * hidden_state], dim=1))
        hidden_state = (1 - update_value) * hidden_state + update_value * hidden_candidate
        return hidden_state


class GateUnitExpand2(nn.Module):
    """
    拓展的gate单元，支持自由的因素大小。
    """

    def __init__(self, factor_size, hidden_size, div=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_W = nn.Sequential(nn.Linear(self.hidden_size * 2+factor_size, self.hidden_size),
                                  nn.Sigmoid())
        self.hidden_trans = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.Tanh()
        )
        self.div = div
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, message: torch.Tensor, factor: torch.Tensor, hidden_state: torch.Tensor)->torch.Tensor:
        """
        通过类似GRU的门控机制更新实体表示

        :param message: message[batch_size,input_size]
        :param query_r: query_r[batch_size,input_size]
        :param hidden_state: if it is none,it will be allocated a zero tensor hidden state
        :return:
        """
        # 计算门, 计算门时考虑到查询的关系
        reset_value = self.gate_W(torch.cat([message, factor, hidden_state], dim=1))
        # 计算候选隐藏表示
        hidden_candidate = self.hidden_trans(torch.cat([reset_value *message,  hidden_state], dim=1))+hidden_state
        return hidden_candidate