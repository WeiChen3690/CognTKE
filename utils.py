from warnings import WarningMessage
import torch
from scipy.sparse import csr_matrix
import numpy as np
from scipy.stats import rankdata
import copy
import subprocess
from typing import Optional, List, Dict
from collections import defaultdict
import logging
import os
import re
from scipy.sparse import csr_matrix
import concurrent.futures
import itertools
import multiprocessing


def get_logger(log_filename: str):
    """指定保存日志的文件路径，日志级别，以及调用文件
        将日志存入到指定的文件中
        :paramlogger:
        """
    # 创建一个logger
    logger = logging.getLogger(log_filename)
    logger.setLevel(logging.INFO)
    # 此处的判断是为了不重复调用test_log，导致重复打印出日志；第一次调用就会创建一个，第二次就不会再次调用了，也就不会出现重复日志的情况

    # 创建一个handler，用于写入日志文件
    if len(log_filename) == 0 or log_filename[-1] in ('/', '\\'):
        raise FileNotFoundError("无效的log文件地址,请使用文件而不是目录当作log输出地")
    father_dir = os.path.dirname(os.path.abspath(log_filename))
    if not os.path.exists(father_dir):
        os.makedirs(father_dir, exist_ok=True)
    fh = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    # 创建一个hander用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # 定义handler的输出格式
    formeter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] %(message)s')
    fh.setFormatter(formeter)
    ch.setFormatter(formeter)
    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def calc_neighborhood_in_subgraph(idx, step, fact_sub_matrix, graph_extended, node_1hot):
    edge_1hot = fact_sub_matrix.dot(node_1hot)
    # (事实，查询次序)对
    edges = np.nonzero(edge_1hot)
    # 表示三元组和目标查询三元组的时间差（相对时间）,不过0留给了等价关系，表示无意义，没有时间概念
    timestamp = np.ones((edges[1].size, 1)) * (step - idx)
    return np.concatenate([np.expand_dims(edges[1], 1), graph_extended[edges[0]], timestamp],axis=1)

class Dataloader(object):
    def __init__(self, path):
        self.path = path

    def load_tkg(self):
        self._load()
        self._split_data()
        self._load_graph()

    def _read_dict(self, file_name):
        id_dict = {}
        id_list = []
        with open(self.path + file_name, encoding='utf-8') as file:
            content = file.read()
            content = content.strip()
            content = content.split("\n")
            for line in content:
                line = line.strip()
                line = line.split('\t')
                id_dict[line[0]] = int(line[1])
                id_list.append(line[0])
        return id_dict, id_list

    def _read_fact(self, file_name: str):
        facts = []
        with open(self.path + file_name, encoding='utf-8') as f:
            content = f.read()
            content = content.strip()
            content = content.split("\n")
            for line in content:
                fact = line.split()
                facts.append([int(fact[0]), int(fact[1]), int(fact[2]), fact[3]])
                # reverse
                facts.append([int(fact[2]), int(fact[1]) + self.num_relation, int(fact[0]), fact[3]])
        return facts

    def _load(self):
        self.entity2id, self.id2entity = self._read_dict('entity2id.txt')
        self.num_entity = len(self.id2entity)

        self.relation2id, self.id2relation = self._read_dict('relation2id.txt')
        self.num_relation = len(self.id2relation)

        # reverse
        reverse_rela = []
        for rela in self.id2relation:
            reverse_rela.append(rela + '_reverse')
        for rela_reverse in reverse_rela:
            self.id2relation.append(rela_reverse)

        # self loop
        self.id2relation.append('idd')
        self.num_relation_extended = len(self.id2relation)

        self.data_train = self._read_fact('train.txt')
        self.data_valid = self._read_fact('valid.txt')
        self.data_test = self._read_fact('test.txt')

    def _split_by_time(self, data):
        timeid_list = []
        # 按时间片的filter，格式为self.time_filter[时间id][(s,q)]=[o_1,o_2,o_3]

        for fact in data:
            if fact[3] not in self.time2id.keys():
                self.time2id[fact[3]] = len(self.time2id)
                timeid_list.append(self.time2id[fact[3]])
                self.data_splited.append([])
                self.time_filter[self.time2id[fact[3]]] = defaultdict(list)
            self.data_splited[self.time2id[fact[3]]].append([fact[0], fact[1], fact[2]])
            self.time_filter[self.time2id[fact[3]]][(fact[0], fact[1])].append(fact[2])
        return timeid_list

    def _split_data(self):
        """
        因为时间的连续性问题，按照传统的思路应该是根据前几个时间片来考虑未来，所以有些类似于inductive实验
        问题1：
        不过，这种模式并没有考虑到长期的历史情况。
        比如，
        正面案例1：
        美国打击了巴基斯坦的外汇市场（假如），近期巴基斯坦财政危机，急需国外政府支援。联系了中国，此时巴基斯坦下一步会寻求谁的支援？
        答案是：中国
        反面案例1：
        美国打击了巴基斯坦的外汇市场（假如），近期巴基斯坦财政危机，急需国外政府支援，此时巴基斯坦下一步会寻求谁的支援？
        答案是：美国
        分析：因为近期的国家实体只有美国，所以只能推导出美国。但是如果考虑长期的历史信息（长历史依赖，历史表示）或是静态关系知识，则可以推导出是中国。正确的链条是巴基斯坦和中国友好，所以遇到困难时会找中国。
        问题2：
        历史记忆保留问题，
        推理时只使用了最后一层的实体，而没有考虑更近的实体。
        """
        self.time2id = {}
        self.data_splited = []
        self.time_filter = {}

        self.time_list_train = self._split_by_time(self.data_train)
        self.time_length_train = len(self.time_list_train)

        self.time_list_valid = self._split_by_time(self.data_valid)
        self.time_length_valid = len(self.time_list_valid)

        self.time_list_test = self._split_by_time(self.data_test)
        self.time_length_test = len(self.time_list_test)

        self.time_length = self.time_length_train + self.time_length_valid + self.time_length_test
        # list to narray
        for i in range(self.time_length):
            self.data_splited[i] = np.array(self.data_splited[i], dtype='int64')

    def _load_graph(self):
        self.fact_sub_matrix = []
        self.graph_extended = []
        for i in range(self.time_length):
            KG = self.data_splited[i]
            self.graph_extended.append(KG)
            num_fact = KG.shape[0]
            fsm = csr_matrix((np.ones(num_fact, ), (np.arange(num_fact), KG[:, 0])),
                             shape=(num_fact, self.num_entity))
            self.fact_sub_matrix.append(fsm)

    def get_batch(self, time_stamp, index):
        return self.data_splited[time_stamp][index]

    def get_neighbors(self, nodes, time_stamp):
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])),
                               shape=(self.num_entity, nodes.shape[0]))
        edge_1hot = self.fact_sub_matrix[time_stamp].dot(node_1hot)
        # (edge id,batch_idx)
        edges = np.nonzero(edge_1hot)
        # (batch_idx, head, rela, tail)
        sampled_edges = np.concatenate([np.expand_dims(edges[1], 1), self.graph_extended[time_stamp][edges[0]]], axis=1)

        # 我把预处理时添加的自环边删除了，现在在这添加自环边（和原先的方案自环的覆盖比例不同，这一版能让旧的实体通过自环边前往新的时间片）
        idd = np.vstack([nodes[:, 0], nodes[:, 1], np.ones(shape=(nodes.shape[0])) * (self.num_relation_extended - 1),
                         nodes[:, 1]]).T
        sampled_edges = np.concatenate([sampled_edges, idd], axis=0)

        sampled_edges = torch.LongTensor(sampled_edges).cuda()
        head_nodes, head_index = torch.unique(sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)
        mask = sampled_edges[:, 2] == (self.num_relation * 2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]
        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
        return tail_nodes, sampled_edges, old_nodes_new_idx

    def get_neighbors_in_period(self, nodes, target_timestamp, step):
        """
        获取节点集合在相邻的时间区间的图中的邻居,自由搜索
        edge:  list[id_in_batch,sub,rel,obj,new_sub_id,new_obj_id,timestamp]
        :param nodes:
        :param target_timestamp:
        :param step:
        :return:
        """
        # 假定推理的三元组时间为10，目标时间区间是3，按照现有的策略，检索的时间片为7,8(+1),9(+2)，一共3个，不涉及10.
        # 这一处确保开始时间不会刷到未来去（python的负索引是有效的），避免出现负索引到测试集的情况
        step = min(step, target_timestamp)
        start_time_stamp = max(target_timestamp - step, 0)

        # subject-查询次序矩阵
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])),
                               shape=(self.num_entity, nodes.shape[0]))
        # 事实-查询次序矩阵
        edge_list = []
        for i in range(step):
            edge_1hot = self.fact_sub_matrix[start_time_stamp + i].dot(node_1hot)
            # (事实，查询次序)对
            edges = np.nonzero(edge_1hot)
            # 表示三元组和目标查询三元组的时间差（相对时间）,不过0留给了等价关系，表示无意义，没有时间概念
            timestamp = np.ones((edges[1].size, 1)) * (step - i)
            edge_list.append(
                np.concatenate(
                    [np.expand_dims(edges[1], 1), self.graph_extended[start_time_stamp + i][edges[0]], timestamp],
                    axis=1))

        # 我把预处理时添加的自环边删除了，现在在这添加自环边（和原先的方案自环的覆盖比例不同，这一版能让旧的实体通过自环边前往新的时间片）
        idd = np.vstack([nodes[:, 0], nodes[:, 1], np.ones(shape=(nodes.shape[0])) * (self.num_relation_extended - 1),
                         nodes[:, 1], np.zeros(nodes[:, 1].shape)]).T
        edge_list.append(idd)
        sampled_edges = np.concatenate(edge_list, axis=0)

        sampled_edges = torch.LongTensor(sampled_edges).cuda()
        head_nodes, head_index = torch.unique(sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)
        mask = sampled_edges[:, 2] == (self.num_relation * 2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]
        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
        # 把相对时间放在最后一列
        sampled_edges[:, [4, 5, 6]] = sampled_edges[:, [5, 6, 4]]
        sampled_edges = sampled_edges.contiguous()
        return tail_nodes, sampled_edges, old_nodes_new_idx
    
    def get_neighbors_in_period_concurrent(self, nodes, target_timestamp, step):
        """
        获取节点集合在相邻的时间区间的图中的邻居,自由搜索
        edge:  list[id_in_batch,sub,rel,obj,new_sub_id,new_obj_id,timestamp]
        :param nodes:
        :param target_timestamp:
        :param step:
        :return:
        """
        # 假定推理的三元组时间为10，目标时间区间是3，按照现有的策略，检索的时间片为7,8(+1),9(+2)，一共3个，不涉及10.
        # 这一处确保开始时间不会刷到未来去（python的负索引是有效的），避免出现负索引到测试集的情况
        step = min(step, target_timestamp)
        start_time_stamp = max(target_timestamp - step, 0)

        # subject-查询次序矩阵
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])),
                               shape=(self.num_entity, nodes.shape[0]))
        

        # 创建进程池，包含16个进程
        with concurrent.futures.ProcessPoolExecutor(max_workers=16) as pool:
            # 计算1~10的平方，使用map方法在多个进程中并行计算
            edge_list = list(pool.map(calc_neighborhood_in_subgraph, range(step),itertools.repeat(step), self.fact_sub_matrix[start_time_stamp:start_time_stamp+step], self.graph_extended[start_time_stamp:start_time_stamp+step],itertools.repeat(node_1hot)))

        

        # 我把预处理时添加的自环边删除了，现在在这添加自环边（和原先的方案自环的覆盖比例不同，这一版能让旧的实体通过自环边前往新的时间片）
        idd = np.vstack([nodes[:, 0], nodes[:, 1], np.ones(shape=(nodes.shape[0])) * (self.num_relation_extended - 1),
                         nodes[:, 1], np.zeros(nodes[:, 1].shape)]).T
        edge_list.append(idd)
        sampled_edges = np.concatenate(edge_list, axis=0)

        sampled_edges = torch.LongTensor(sampled_edges).cuda()
        head_nodes, head_index = torch.unique(sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)
        mask = sampled_edges[:, 2] == (self.num_relation * 2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]
        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
        # 把相对时间放在最后一列
        sampled_edges[:, [4, 5, 6]] = sampled_edges[:, [5, 6, 4]]
        sampled_edges = sampled_edges.contiguous()
        return tail_nodes, sampled_edges, old_nodes_new_idx

    def get_subgraph_in_period(self, target_timestamp, window_size):
        """
        获取节点集合在相邻的时间区间的完整子图，用于构建邻域上下文信息
        edge:  list[id_in_batch,sub,rel,obj,new_sub_id,new_obj_id,timestamp]
        :param nodes:
        :param target_timestamp:
        :param step:
        :return:
        """
        # 假定推理的三元组时间为10，目标时间区间是3，按照现有的策略，检索的时间片为7,8(+1),9(+2)，一共3个，不涉及10.
        # 这一处确保开始时间不会刷到未来去（python的负索引是有效的），避免出现负索引到测试集的情况
        window_size = min(window_size, target_timestamp)
        start_time_stamp = max(target_timestamp - window_size, 0)

        # 事实-查询次序矩阵
        edge_list = []
        nodes = []
        for i in range(window_size):
            # (事实，查询次序)对
            edges = self.graph_extended[start_time_stamp + i]
            # 表示三元组和目标查询三元组的时间差（相对时间）,不过0留给了等价关系，表示无意义，没有时间概念
            timestamp = np.ones((edges.shape[0], 1)) * (window_size - i)
            edge_list.append(np.concatenate([edges, timestamp], axis=1))
            nodes.extend(edges[:, 0].tolist())
            nodes.extend(edges[:, 2].tolist())
        # 添加等价关系
        unique_nodes = np.array(list(set(nodes)))
        idd = np.vstack(
            [unique_nodes, np.ones(shape=(unique_nodes.shape[0])) * (self.num_relation_extended - 1), unique_nodes,
             np.zeros(unique_nodes.shape)]).T
        edge_list.append(idd)
        sampled_edges = np.concatenate(edge_list, axis=0)

        sampled_edges = torch.LongTensor(sampled_edges)
        head_nodes, head_index = torch.unique(sampled_edges[:, [0]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:, [2]], dim=0, sorted=True, return_inverse=True)
        mask = sampled_edges[:, 1] == (self.num_relation * 2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]
        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
        # 把相对时间放在最后一列
        sampled_edges[:, [3, 4, 5]] = sampled_edges[:, [4, 5, 3]]
        sampled_edges = sampled_edges.contiguous()
        return tail_nodes, sampled_edges, old_nodes_new_idx

    

    def get_neighbors_in_short_long_period(self, nodes, origin_nodes, target_timestamp, step):
        """
        获取节点集合在相邻的时间区间(短)的图中的邻居,以及永续的时间区间（从最开始到当前阶段）的小范围，只能用于最后一层的邻居采集
        edge:  list[id_in_batch,sub,rel,obj,new_sub_id,new_obj_id,timestamp]
        :param nodes:
        :param target_timestamp:
        :param step:
        :return:
        """

        step = min(step, target_timestamp)
        start_time_stamp = max(target_timestamp - step, 0)

        # 假定推理的三元组时间为10，目标时间区间是3，按照现有的策略，检索的时间片为7,8(+1),9(+2)，一共3个，不涉及10.
        start_time_stamp = target_timestamp - step
        # subject-查询次序矩阵
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])),
                               shape=(self.num_entity, nodes.shape[0]))
        # 事实-查询次序矩阵
        edge_list = []
        for i in range(step):
            edge_1hot = self.fact_sub_matrix[start_time_stamp + i].dot(node_1hot)
            # (事实，查询次序)对
            edges = np.nonzero(edge_1hot)
            # 表示三元组和目标查询三元组的时间差（相对时间）,不过0留给了等价关系，表示无意义，没有时间概念
            # 比如，距离查询时间距离为5的时间戳为
            timestamp = np.ones((edges[1].size, 1)) * (step - i)
            edge_list.append(
                np.concatenate(
                    [np.expand_dims(edges[1], 1), self.graph_extended[start_time_stamp + i][edges[0]], timestamp],
                    axis=1))

        # 我把预处理时添加的自环边删除了，现在在这添加自环边（和原先的方案自环的覆盖比例不同，这一版能让旧的实体通过自环边前往新的时间片）
        idd = np.vstack([nodes[:, 0], nodes[:, 1], np.ones(shape=(nodes.shape[0])) * (self.num_relation_extended - 1),
                         nodes[:, 1], np.zeros(nodes[:, 1].shape)]).T
        edge_list.append(idd)

        start_time_stamp = 0
        # 开始添加完整历史的交互
        # 设置原始的节点的稀疏矩阵
        origin_node_1hot = csr_matrix((np.ones(len(origin_nodes)), (origin_nodes[:, 1], origin_nodes[:, 0])),
                                      shape=(self.num_entity, origin_nodes.shape[0]))
        # 一共需要添加target_timestamp-step个时间片的数据
        for i in range(target_timestamp - step):
            edge_1hot = self.fact_sub_matrix[i].dot(origin_node_1hot)
            # (事实，查询次序)对
            edges = np.nonzero(edge_1hot)
            # 表示三元组和目标查询三元组的时间差（相对时间）,不过0留给了等价关系，表示无意义，没有时间概念
            timestamp = np.ones((edges[1].size, 1)) * (target_timestamp - i)
            edge_list.append(
                np.concatenate([np.expand_dims(edges[1], 1), self.graph_extended[i][edges[0]], timestamp], axis=1))

        sampled_edges = np.concatenate(edge_list, axis=0)

        sampled_edges = torch.LongTensor(sampled_edges).cuda()
        head_nodes, head_index = torch.unique(sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)
        mask = sampled_edges[:, 2] == (self.num_relation * 2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]
        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
        # 把相对时间放在最后一列
        sampled_edges[:, [4, 5, 6]] = sampled_edges[:, [5, 6, 4]]
        sampled_edges = sampled_edges.contiguous()
        return tail_nodes, sampled_edges, old_nodes_new_idx

    def get_neighbors_in_short_limited_long_period(self, nodes, origin_nodes, target_timestamp, step, long_limited):
        """
        获取节点集合在相邻的时间区间(短)的图中的邻居,以及永续的时间区间（从最开始到当前阶段）的小范围，只能用于最后一层的邻居采集
        edge:  list[id_in_batch,sub,rel,obj,new_sub_id,new_obj_id,timestamp]
        :param nodes:
        :param target_timestamp:
        :param step:
        :return:
        """

        step = min(step, target_timestamp)
        start_time_stamp = max(target_timestamp - step, 0)

        # 假定推理的三元组时间为10，目标时间区间是3，按照现有的策略，检索的时间片为7,8(+1),9(+2)，一共3个，不涉及10.
        start_time_stamp = target_timestamp - step
        # subject-查询次序矩阵
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])),
                               shape=(self.num_entity, nodes.shape[0]))
        # 事实-查询次序矩阵
        edge_list = []
        for i in range(step):
            edge_1hot = self.fact_sub_matrix[start_time_stamp + i].dot(node_1hot)
            # (事实，查询次序)对
            edges = np.nonzero(edge_1hot)
            # 表示三元组和目标查询三元组的时间差（相对时间）,不过0留给了等价关系，表示无意义，没有时间概念
            # 比如，距离查询时间距离为5的时间戳为
            timestamp = np.ones((edges[1].size, 1)) * (step - i)
            edge_list.append(
                np.concatenate(
                    [np.expand_dims(edges[1], 1), self.graph_extended[start_time_stamp + i][edges[0]], timestamp],
                    axis=1))

        # 我把预处理时添加的自环边删除了，现在在这添加自环边（和原先的方案自环的覆盖比例不同，这一版能让旧的实体通过自环边前往新的时间片）
        idd = np.vstack([nodes[:, 0], nodes[:, 1], np.ones(shape=(nodes.shape[0])) * (self.num_relation_extended - 1),
                         nodes[:, 1], np.zeros(nodes[:, 1].shape)]).T
        edge_list.append(idd)

        start_time_stamp = 0
        # 开始添加完整历史的交互
        # 设置原始的节点的稀疏矩阵
        origin_node_1hot = csr_matrix((np.ones(len(origin_nodes)), (origin_nodes[:, 1], origin_nodes[:, 0])),
                                      shape=(self.num_entity, origin_nodes.shape[0]))
        # 一共需要添加target_timestamp-step个时间片的数据
        start_time_stamp = target_timestamp - step  # 从更新的时间往更古老的时间追溯
        for i in range(min(max(target_timestamp - step, 0), long_limited)):
            edge_1hot = self.fact_sub_matrix[start_time_stamp - i].dot(origin_node_1hot)
            # (事实，查询次序)对
            edges = np.nonzero(edge_1hot)
            # 表示三元组和目标查询三元组的时间差（相对时间）,不过0留给了等价关系，表示无意义，没有时间概念
            timestamp = np.ones((edges[1].size, 1)) * (step + i)
            edge_list.append(np.concatenate(
                [np.expand_dims(edges[1], 1), self.graph_extended[start_time_stamp - i][edges[0]], timestamp], axis=1))

        sampled_edges = np.concatenate(edge_list, axis=0)

        sampled_edges = torch.LongTensor(sampled_edges).cuda()
        head_nodes, head_index = torch.unique(sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)
        mask = sampled_edges[:, 2] == (self.num_relation * 2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]
        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
        # 把相对时间放在最后一列
        sampled_edges[:, [4, 5, 6]] = sampled_edges[:, [5, 6, 4]]
        sampled_edges = sampled_edges.contiguous()
        return tail_nodes, sampled_edges, old_nodes_new_idx

    def get_neighbors_in_short_long_period_seperately(self, nodes, origin_nodes, target_timestamp, step):
        """
        获取节点集合在相邻的时间区间(短)的图中的邻居,以及永续的时间区间（从最开始到当前阶段）的小范围，只能用于最后一层的邻居采集,分离了全局的边和局部的边。
        edge:  list[id_in_batch,sub,rel,obj,new_sub_id,new_obj_id,timestamp]
        :param nodes:
        :param target_timestamp:
        :param step:
        :return: tail_nodes, old_nodes_new_idx, local edges, global edges
        """

        step = min(step, target_timestamp)
        start_time_stamp = max(target_timestamp - step, 0)

        # 假定推理的三元组时间为10，目标时间区间是3，按照现有的策略，检索的时间片为7,8(+1),9(+2)，一共3个，不涉及10.
        start_time_stamp = target_timestamp - step
        # subject-查询次序矩阵
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])),
                               shape=(self.num_entity, nodes.shape[0]))
        # 事实-查询次序矩阵
        edge_list = []
        for i in range(step):
            edge_1hot = self.fact_sub_matrix[start_time_stamp + i].dot(node_1hot)
            # (事实，查询次序)对
            edges = np.nonzero(edge_1hot)
            # 表示三元组和目标查询三元组的时间差（相对时间）,不过0留给了等价关系，表示无意义，没有时间概念
            # 比如，距离查询时间距离为5的时间戳为
            timestamp = np.ones((edges[1].size, 1)) * (step - i)
            edge_list.append(
                np.concatenate(
                    [np.expand_dims(edges[1], 1), self.graph_extended[start_time_stamp + i][edges[0]], timestamp],
                    axis=1))

        # 我把预处理时添加的自环边删除了，现在在这添加自环边（和原先的方案自环的覆盖比例不同，这一版能让旧的实体通过自环边前往新的时间片）
        idd = np.vstack([nodes[:, 0], nodes[:, 1], np.ones(shape=(nodes.shape[0])) * (self.num_relation_extended - 1),
                         nodes[:, 1], np.zeros(nodes[:, 1].shape)]).T
        edge_list.append(idd)
        short_count = sum([len(edge) for edge in edge_list])
        start_time_stamp = 0
        # 开始添加完整历史的交互
        # 设置原始的节点的稀疏矩阵
        origin_node_1hot = csr_matrix((np.ones(len(origin_nodes)), (origin_nodes[:, 1], origin_nodes[:, 0])),
                                      shape=(self.num_entity, origin_nodes.shape[0]))
        # 一共需要添加target_timestamp-step个时间片的数据
        for i in range(target_timestamp):
            edge_1hot = self.fact_sub_matrix[i].dot(origin_node_1hot)
            # (事实，查询次序)对
            edges = np.nonzero(edge_1hot)
            # 表示三元组和目标查询三元组的时间差（相对时间）,不过0留给了等价关系，表示无意义，没有时间概念
            timestamp = np.ones((edges[1].size, 1)) * (target_timestamp - i)
            edge_list.append(
                np.concatenate([np.expand_dims(edges[1], 1), self.graph_extended[i][edges[0]], timestamp], axis=1))

        sampled_edges = np.concatenate(edge_list, axis=0)

        sampled_edges = torch.LongTensor(sampled_edges).cuda()
        head_nodes, head_index = torch.unique(sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)
        mask = sampled_edges[:, 2] == (self.num_relation * 2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]
        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
        # 把相对时间放在最后一列
        sampled_edges[:, [4, 5, 6]] = sampled_edges[:, [5, 6, 4]]
        sampled_edges = sampled_edges.contiguous()
        return tail_nodes, old_nodes_new_idx, sampled_edges[:short_count], sampled_edges[short_count:]

    def get_neighbors_with_visit(self, nodes, time_stamp):
        """返回值增加了访问记录

        Parameters
        ----------
        nodes : _type_
            _description_
        time_stamp : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])),
                               shape=(self.num_entity, nodes.shape[0]))
        edge_1hot = self.fact_sub_matrix[time_stamp].dot(node_1hot)
        # (edge id,batch_idx)
        edges = np.nonzero(edge_1hot)
        # (batch_idx, head, rela, tail)
        sampled_edges = np.concatenate([np.expand_dims(edges[1], 1), self.graph_extended[time_stamp][edges[0]]],
                                       axis=1)
        latest_visited_nodes = torch.unique(torch.from_numpy(sampled_edges[:, [0, 3]]), dim=0, sorted=True)

        # 我把预处理时添加的自环边删除了，现在在这添加自环边（和原先的方案自环的覆盖比例不同，这一版能让旧的实体通过自环边前往新的时间片）
        idd = np.vstack([nodes[:, 0], nodes[:, 1], np.ones(shape=(nodes.shape[0])) * (self.num_relation_extended - 1),
                         nodes[:, 1]]).T
        sampled_edges = np.concatenate([sampled_edges, idd], axis=0)

        sampled_edges = torch.LongTensor(sampled_edges).cuda()
        head_nodes, head_index = torch.unique(sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)
        mask = sampled_edges[:, 2] == (self.num_relation * 2)
        _, old_idx = head_index[mask].sort()

        old_nodes_new_idx = tail_index[mask][old_idx]
        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
        return tail_nodes, sampled_edges, old_nodes_new_idx, latest_visited_nodes


class ElasticEmbedding:
    """"
    用于组织稀疏张量的数据结构，用于长期记忆出现过的实体的表示。
    """

    def __init__(self, batch_size: int, entity_num: int, hidden_size: int, device):
        self.index_matrix = - np.ones(shape=(batch_size, entity_num), dtype=np.int64)
        self.hidden_size = hidden_size
        # 关于数据是否需要设置可反向传播
        self.data = torch.zeros(batch_size, hidden_size).to(device)
        self.device = device

    def get(self, index: np.ndarray) -> torch.Tensor:
        """
        按照索引获取节点的隐藏表示
        :param index: array(list) of [sample_id, entity_id]
        :return: Tensor, size=(len(index), hidden_size),获取的数据
        """
        _index = self.index_matrix[index[:, 0], index[:, 1]]
        effective_mask = _index > -1
        effective_index = _index[effective_mask]
        temp = torch.zeros(len(index), self.hidden_size).to(self.device)
        temp[effective_mask] = self.data[effective_index]
        return temp

    def set(self, index: np.ndarray, data: torch.Tensor) -> None:
        """
        将节点的表示更新到包含所有的节点表示的表中
        :param index:
        :param data:D
        :return:
        """
        _index = self.index_matrix[index[:, 0], index[:, 1]]
        effective_mask = _index > -1
        missing_mask = effective_mask == False

        # 先创建缺失的索引
        missing_index = index[missing_mask]
        self.index_matrix[missing_index[:, 0], missing_index[:, 1]] = \
            np.arange(len(self.data), len(self.data) + len(missing_index))

        effective_index = _index[effective_mask]
        # 能找到的直接赋值操作
        self.data[effective_index] = data[effective_mask]
        # 找不到的
        missing_data = data[missing_mask]
        self.data = torch.vstack((self.data, missing_data))

    def get_all(self):
        valid_mask = self.index_matrix > -1
        indices = self.index_matrix[valid_mask]
        data = self.data[indices]
        return *valid_mask.nonzero(), data


class SparseEmbedding2D:
    """
    可以进行索引的稀疏矩阵，这里专用于存储历史的查询相关的表示，即根据查询头实体、查询关系、可能的答案实体，存储或读取相应的历史的表示（不参与反向传播）。考虑到效率和空间问题，而且时间的推理是从前往后进行推理的，这里存储的表示是不考虑时间的。
    在测试时发现，csr_matrix只支持小于2维的稀疏矩阵，不支持大于2维的稀疏矩阵，所以这个数据结构还得改……
    先通过一个2维的稀疏索引矩阵将头实体和关系的组合索引到虚拟的组合编号，再通过组合的编号和尾实体索引到最后的表示的编号。
    """

    def __init__(self, max_index_shape: tuple, hidden_size: int, device):
        self.index_matrix = csr_matrix(max_index_shape, dtype=np.int64)  # 无效的索引值为0,为了带有区分性，里面存储的索引值比实际的大1
        self.index_dim = len(max_index_shape)
        if self.index_dim not in (2, 3):
            raise Exception(f'不支持当前传入的索引维度，当前为{self.index_dim}，请当前只支持2,3')
        self.hidden_size = hidden_size
        # 设置初始的数据存储区块
        self.data = torch.zeros(0, hidden_size).to(device)
        self.device = device

    def parse_index(self, index: np.ndarray):
        if index.shape[1] != self.index_dim:
            raise Warning(f'当前传入的索引维度数为{index.shape[1]},和初始化时定义的数值{self.index_dim == 2}不匹配，请检查是否有问题')
        if self.index_dim == 2:
            _index = self.index_matrix[index[:, 0], index[:, 1]]
        else:
            raise Exception(f'不支持当前传入的索引维度，当前为{self.index_dim}，请当前只支持2')
        # 里面存储的索引值比实际的大1,所以取出来后需要再减去1
        _index = np.array(_index) - 1
        effective_mask = _index > -1
        effective_index = _index[effective_mask]
        return _index, effective_mask, effective_index

    def get(self, index: np.ndarray) -> torch.Tensor:
        """
        按照索引获取节点的隐藏表示
        :param index: array(list) of [dim1, dim2]
        :return: Tensor, size=(len(index), hidden_size),获取的数据
        """
        _index, effective_mask, effective_index = self.parse_index(index)

        temp = torch.zeros(len(index), self.hidden_size).to(self.device)
        temp[effective_mask] = self.data[effective_index]
        return temp

    def get_valid_triple_indexes(self, subjects: np.ndarray, relations: np.ndarray):
        """
        这个方法限制了只能知道头实体和关系，查找所有有效的尾实体的表示的索引。
        """
        new_subjects, cols = self.index_matrix[subjects, relations].nonzero()
        self.index_matrix[subjects, relations]

    def set(self, index: np.ndarray, data: torch.Tensor) -> None:
        """
        将节点的表示更新到包含所有的节点表示的表中
        :param index:
        :param data:
        :return:
        """

        assert index.shape[0] == data.shape[0], f'index的长度{index.shape[0]}必须和data的长度{data.shape[0]}一致'

        _index, effective_mask, effective_index = self.parse_index(index)

        missing_mask = effective_mask == False

        # 先创建缺失的索引,指向数据块中的位置
        missing_index = index[missing_mask]
        # 里面存储的索引值比实际的大1,所以向里面存储数字时需要先加1
        if self.index_dim == 2:
            self.index_matrix[index[:, 0], index[:, 1]] = \
                np.arange(len(self.data) + 1, len(self.data) + len(missing_index) + 1)
        else:
            raise Exception(f'不支持当前传入的索引维度，当前为{self.index_dim}，请当前只支持2,3,4')
        data = data.detach()
        # 能找到的直接赋值操作
        self.data[effective_index] = data[effective_mask]
        # 找不到的
        missing_data = data[missing_mask]
        self.data = torch.vstack((self.data, missing_data))


class SparseEmbedding3D_failed:
    """
    可以进行索引的稀疏矩阵，这里专用于存储历史的查询相关的表示，即根据查询头实体、查询关系、可能的答案实体，存储或读取相应的历史的表示（不参与反向传播）。考虑到效率和空间问题，而且时间的推理是从前往后进行推理的，这里存储的表示是不考虑时间的。
    在测试时发现，csr_matrix只支持小于2维的稀疏矩阵，不支持大于2维的稀疏矩阵，所以这个数据结构还得改……
    先通过一个2维的稀疏索引矩阵将头实体和关系的组合索引到虚拟的组合编号，再通过组合的编号和尾实体索引到最后的表示的编号。
    """

    def __init__(self, max_index_shape: tuple, hidden_size: int, device):
        self.index_matrix = csr_matrix(max_index_shape, dtype=np.int64)  # 无效的索引值为0,为了带有区分性，里面存储的索引值比实际的大1
        self.index_dim = len(max_index_shape)
        if self.index_dim != 3:
            raise Exception(f'不支持当前传入的索引维度，当前为{self.index_dim}，请当前只支持3')
        self.hidden_size = hidden_size
        # 设置初始的数据存储区块
        self.data = torch.zeros(0, hidden_size).to(device)
        self.device = device
        self.query2index = []

    def parse_index(self, index: np.ndarray):
        if index.shape[1] != self.index_dim:
            raise Warning(f'当前传入的索引维度数为{index.shape[1]},和初始化时定义的数值{self.index_dim == 2}不匹配，请检查是否有问题')

        if self.index_dim == 3:
            _index = self.index_matrix[index[:, 0], index[:, 1], index[:, 2]]
        else:
            raise Exception(f'不支持当前传入的索引维度，当前为{self.index_dim}，请当前只支持2,3,4')
        # 里面存储的索引值比实际的大1,所以取出来后需要再减去1
        _index = np.array(_index) - 1
        effective_mask = _index > -1
        effective_index = _index[effective_mask]
        return _index, effective_mask, effective_index

    def get(self, index: np.ndarray) -> torch.Tensor:
        """
        按照索引获取节点的隐藏表示
        :param index: array(list) of [dim1, dim2]
        :return: Tensor, size=(len(index), hidden_size),获取的数据
        """
        _index, effective_mask, effective_index = self.parse_index(index)

        temp = torch.zeros(len(index), self.hidden_size).to(self.device)
        temp[effective_mask] = self.data[effective_index]
        return temp

    def get_valid_triple_indexes(self, subjects: np.ndarray, relations: np.ndarray):
        """
        这个方法限制了只能知道头实体和关系，查找所有有效的尾实体的表示的索引。
        """
        new_subjects, cols = self.index_matrix[subjects, relations].nonzero()
        self.index_matrix[subjects, relations]

    def set(self, index: np.ndarray, data: torch.Tensor) -> None:
        """
        将节点的表示更新到包含所有的节点表示的表中
        :param index:
        :param data:
        :return:
        """

        assert index.shape[0] == data.shape[0], f'index的长度{index.shape[0]}必须和data的长度{data.shape[0]}一致'

        _index, effective_mask, effective_index = self.parse_index(index)

        missing_mask = effective_mask == False

        # 先创建缺失的索引,指向数据块中的位置
        missing_index = index[missing_mask]
        # 里面存储的索引值比实际的大1,所以向里面存储数字时需要先加1
        if self.index_dim == 2:
            self.index_matrix[index[:, 0], index[:, 1]] = \
                np.arange(len(self.data) + 1, len(self.data) + len(missing_index) + 1)
        elif self.index_dim == 3:
            self.index_matrix[index[:, 0], index[:, 1], index[:, 2]] = \
                np.arange(len(self.data) + 1, len(self.data) + len(missing_index) + 1)
        elif self.index_dim == 4:
            self.index_matrix[index[:, 0], index[:, 1], index[:, 2], index[:, 3]] = \
                np.arange(len(self.data) + 1, len(self.data) + len(missing_index) + 1)
        else:
            raise Exception(f'不支持当前传入的索引维度，当前为{self.index_dim}，请当前只支持2,3,4')
        data = data.detach()
        # 能找到的直接赋值操作
        self.data[effective_index] = data[effective_mask]
        # 找不到的
        missing_data = data[missing_mask]
        self.data = torch.vstack((self.data, missing_data))

    def get_all(self):
        raise NotImplementedError()
        valid_mask = self.index_matrix > -1
        indices = self.index_matrix[valid_mask]
        data = self.data[indices]
        return *valid_mask.nonzero(), data


class SimpleSparseEmbedding3D:
    """
    简陋版的离散张量矩阵，使用字典完成索引，速度可能会比较慢，但是实现简单.数据存储在内存中。
    """

    def __init__(self, ):
        # 设置初始的数据存储区块
        self.dict2 = defaultdict(dict)

    def set(self, indexes: np.ndarray, hiddens):
        """
        indexes: 3维数组，s,r,o
        """
        for triple, hidden in zip(indexes, hiddens.cpu().detach()):
            self.dict2[(triple[0], triple[1])][triple[2]] = hidden

    def get(self, group_ids: np.ndarray, subjects: np.ndarray, relations: np.ndarray):
        new_indexes = []
        new_hiddens = []
        for group_id, subject, relation in zip(group_ids, subjects, relations):
            query_dict = self.dict2.get((subject, relation), None)
            if query_dict is None:
                continue
            for obj, hidden in query_dict.items():
                new_indexes.append((group_id, subject, relation, obj))
                new_hiddens.append(hidden)
        if len(new_indexes) > 0:
            return np.stack(new_indexes, axis=0), torch.stack(new_hiddens, dim=0)
        else:
            return None, None


class NewSparseEmbedding3D:
    def __init__(self, entity_num, relation_num, hidden_dim):
        self.data = torch.zeros(1, hidden_dim)
        self.index = torch.sparse_coo_tensor(size=(entity_num * relation_num, entity_num), dtype=torch.long)
        self.entity_num = entity_num
        self.relation_num = relation_num

    def is_exist(self, raw_index, col_index):
        index = torch.index_select(self.index, dim=0, index=torch.tensor(raw_index, dtype=torch.long)).to_dense()
        index = index[torch.arange(len(raw_index)), torch.tensor(col_index, dtype=torch.long)]
        return ~(index == torch.zeros(index.shape[0])), index

    def set(self, indexes: np.ndarray, hiddens):
        raw_index = indexes[:, 0] * self.relation_num + indexes[:, 1]
        col_index = indexes[:, 2]
        exist, index = self.is_exist(raw_index, col_index)

        # 添加新表示
        if (~exist).sum() > 0:
            i = torch.tensor(np.array([raw_index[~exist], col_index[~exist]]))
            v = torch.arange(self.data.shape[0], self.data.shape[0] + i.shape[1])
            self.index.add_(torch.sparse_coo_tensor(i, v, size=(self.entity_num * self.relation_num, self.entity_num),
                                                    dtype=torch.long))

            self.data = torch.cat([self.data, hiddens[~exist]], dim=0)
        # 覆盖旧表示
        if exist.sum() > 0:
            x = hiddens[exist]
            y = index[exist]
            self.data[index[exist]] = hiddens[exist]

    def get(self, group_ids: np.ndarray, subjects: np.ndarray, relations: np.ndarray):
        row_index = subjects * self.relation_num + relations
        matrix = torch.index_select(self.index, dim=0, index=torch.LongTensor(row_index)).coalesce()
        num = torch.sign(matrix).sum(dim=1).to_dense()
        if num.sum() == 0:
            return None, None
        index = matrix.values()
        ids = torch.repeat_interleave(torch.tensor(group_ids), num).numpy()
        sub = torch.repeat_interleave(torch.tensor(subjects), num).numpy()
        rela = torch.repeat_interleave(torch.tensor(relations), num).numpy()
        obj = matrix.indices()[1].numpy()
        return np.stack([ids, sub, rela, obj], axis=1), torch.stack(list(self.data[index]), dim=0)


class SparseEmbedding3D(ElasticEmbedding):
    """
    基于离散张量和伪哈希表的3维离散张量
    """

    def __init__(self, batch_size: int, entity_num: int, hidden_size: int, device):
        super().__init__(batch_size, entity_num, hidden_size, device)
        self.shape = (2, 5, 2)

    def parse_index_322(self, index3d: np.ndarray):
        """
        将三维索引转换为二维
        """
        dim1 = index3d[:, 0] * self.shape[0] + index3d[:, 1]
        dim2 = index3d[:, 2]
        final_dim = np.concatenate([dim1, dim2], axis=1)
        return final_dim

    def set(self, indexes: np.ndarray, hiddens):
        """
        indexes: 3维数组，s,r,o
        """
        new_indices = self.parse_index_322(indexes)
        # 这里scipy的sp不能放张量，能放张量的又不能修改……
        super().set(new_indices, hiddens)

    def get(self, group_ids: np.ndarray, subjects: np.ndarray, relations: np.ndarray) -> torch.Tensor:
        """
        按照索引获取节点的隐藏表示
        :param index: array(list) of [sample_id, entity_id]
        :return: Tensor, size=(len(index), hidden_size),获取的数据
        """
        dim1 = subjects * self.shape[0] + relations

        _index = self.index_matrix[dim1, :]
        effective_mask = _index > 0
        effective_index = _index[effective_mask] - 1
        # 对应了最后一维
        _index.nonzero()
        return self.data[effective_index]


def gpu_setting(gpu=-1):
    if gpu == -1:
        try:
            gpu = select_gpu()
        except UnicodeDecodeError:
            gpu = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        print('gpu:', gpu)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    return device, gpu


def cal_ranks(scores, labels, filters):
    scores = scores - np.min(scores, axis=1, keepdims=True) + 1e-6
    full_rank = rankdata(-scores, method='ordinal', axis=1)
    filter_scores = scores * filters
    filter_rank = rankdata(-filter_scores, method='ordinal', axis=1)
    mask = np.zeros_like(scores)
    mask[np.arange(mask.shape[0]), labels] = 1
    ranks = (full_rank - filter_rank + 1) * mask
    ranks = ranks[np.nonzero(ranks)]
    return ranks.tolist()


def cal_performance(ranks):
    mrr = (1. / ranks).sum() / len(ranks)
    h_1 = sum(ranks <= 1) * 1.0 / len(ranks)
    h_3 = sum(ranks <= 3) * 1.0 / len(ranks)
    h_10 = sum(ranks <= 10) * 1.0 / len(ranks)
    h_100 = sum(ranks <= 100) * 1.0 / len(ranks)
    return mrr, h_1, h_3, h_10, h_100


def select_gpu():
    try:
        nvidia_info = subprocess.run(
            'nvidia-smi', stdout=subprocess.PIPE).stdout.decode()
    except UnicodeDecodeError:
        nvidia_info = subprocess.run(
            'nvidia-smi', stdout=subprocess.PIPE).stdout.decode("gbk")
    used_list = re.compile(r"(\d+)MiB\s+/\s+\d+MiB").findall(nvidia_info)
    used = [(idx, int(num)) for idx, num in enumerate(used_list)]
    sorted_used = sorted(used, key=lambda x: x[1])
    return sorted_used[0][0]


class EnhancedDict(dict):
    def __init__(__self, *args, **kwargs):
        object.__setattr__(__self, '__parent', kwargs.pop('__parent', None))
        object.__setattr__(__self, '__key', kwargs.pop('__key', None))
        object.__setattr__(__self, '__frozen', False)
        for arg in args:
            if not arg:
                continue
            elif isinstance(arg, dict):
                for key, val in arg.items():
                    __self[key] = __self._hook(val)
            elif isinstance(arg, tuple) and (not isinstance(arg[0], tuple)):
                __self[arg[0]] = __self._hook(arg[1])
            else:
                for key, val in iter(arg):
                    __self[key] = __self._hook(val)

        for key, val in kwargs.items():
            __self[key] = __self._hook(val)

    def __setattr__(self, name, value):
        if hasattr(self.__class__, name):
            raise AttributeError("'Dict' object attribute "
                                 "'{0}' is read-only".format(name))
        else:
            self[name] = value

    def __setitem__(self, name, value):
        isFrozen = (hasattr(self, '__frozen') and
                    object.__getattribute__(self, '__frozen'))
        if isFrozen and name not in super(EnhancedDict, self).keys():
            raise KeyError(name)
        super(EnhancedDict, self).__setitem__(name, value)
        try:
            p = object.__getattribute__(self, '__parent')
            key = object.__getattribute__(self, '__key')
        except AttributeError:
            p = None
            key = None
        if p is not None:
            p[key] = self
            object.__delattr__(self, '__parent')
            object.__delattr__(self, '__key')

    def __add__(self, other):
        if not self.keys():
            return other
        else:
            self_type = type(self).__name__
            other_type = type(other).__name__
            msg = "unsupported operand type(s) for +: '{}' and '{}'"
            raise TypeError(msg.format(self_type, other_type))

    @classmethod
    def _hook(cls, item):
        if isinstance(item, dict):
            return cls(item)
        elif isinstance(item, (list, tuple)):
            return type(item)(cls._hook(elem) for elem in item)
        return item

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __missing__(self, name):
        if object.__getattribute__(self, '__frozen'):
            raise KeyError(name)
        return self.__class__(__parent=self, __key=name)

    def __delattr__(self, name):
        del self[name]

    def to_dict(self):
        base = {}
        for key, value in self.items():
            if isinstance(value, type(self)):
                base[key] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                base[key] = type(value)(
                    item.to_dict() if isinstance(item, type(self)) else
                    item for item in value)
            else:
                base[key] = value
        return base

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        other = self.__class__()
        memo[id(self)] = other
        for key, value in self.items():
            other[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return other

    def update(self, *args, **kwargs):
        other = {}
        if args:
            if len(args) > 1:
                raise TypeError()
            other.update(args[0])
        other.update(kwargs)
        for k, v in other.items():
            if ((k not in self) or
                    (not isinstance(self[k], dict)) or
                    (not isinstance(v, dict))):
                self[k] = v
            else:
                self[k].update(v)

    def __getnewargs__(self):
        return tuple(self.items())

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)

    def __or__(self, other):
        if not isinstance(other, (EnhancedDict, dict)):
            return NotImplemented
        new = EnhancedDict(self)
        new.update(other)
        return new

    def __ror__(self, other):
        if not isinstance(other, (EnhancedDict, dict)):
            return NotImplemented
        new = EnhancedDict(other)
        new.update(self)
        return new

    def __ior__(self, other):
        self.update(other)
        return self

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        else:
            self[key] = default
            return default

    def freeze(self, shouldFreeze=True):
        object.__setattr__(self, '__frozen', shouldFreeze)
        for key, val in self.items():
            if isinstance(val, EnhancedDict):
                val.freeze(shouldFreeze)

    def unfreeze(self):
        self.freeze(False)
