from copy import copy
import models
import utils
import torch
from tqdm import tqdm
import numpy as np
import sys
from utils import get_logger
import json
import time
from models import load_model
import pickle
import os
from utils import EnhancedDict
from typing import Optional,Callable
from datetime import datetime  
import subprocess



class Trainer(object):
    def __init__(self, opts:Optional[EnhancedDict]=None,reevaluate_path:Optional[str]=None,):
        
        if reevaluate_path is not None and opts is None:
            with open(reevaluate_path,"r") as f:
                opts = json.load(f)
                opts = EnhancedDict(opts)
        elif reevaluate_path is None and opts is not None:
            pass
        else:
            raise Exception("参数不能全部是空或全部提供值")
        self.opts = opts
        self.n_layer = opts.n_layer
        self.data = utils.Dataloader(opts.path)
        self.batch_size = opts.batch_size
        self.data.load_tkg()
        self.model_name = opts.model_name
        # 指定评估方法
        evaluate_dict = {
            "single_step":self.evaluate_with_single_steps,
            "multi_step":self.evaluate_with_multi_steps
        }
        self.evaluate = evaluate_dict[opts.get("evaluate_method","single_step")]

                
        # self.tred_gnn = model_dict[self.model_name](self.data, opts)
        self.tred_gnn = load_model(self.model_name)(self.data, opts)
        self.tred_gnn.cuda()
        self.optimizer = torch.optim.Adam(self.tred_gnn.parameters(), lr=opts.lr, weight_decay=opts.lamb)
        self.train_history = []
        self.loss_history = []
        if opts.tag is None or len(opts.tag)==0:
            self.result_dir = f"results/{opts.model_name}/{opts.dataset}/{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        else:
            self.result_dir = f"results/{opts.model_name}/{opts.tag}/{opts.dataset}/{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        self.logger = get_logger(self.result_dir+"/log.txt")
        self.now_epoch=0
        self.best_mrr = 0.
        self.best_result = {}
        self.logger.info("-"*20+"开始"+"-"*20)
        self.logger.info(json.dumps(opts))
        if not isinstance(reevaluate_path,str) :
            with open(f"{self.result_dir}/config.json","w") as f:
                json.dump(self.opts.to_dict(),f)

    def train_epoch(self):
        self.now_epoch+=1
        self.logger.info(f"Start epoch {self.now_epoch} train")
        if self.data.time_length_train - self.n_layer < 0:
            raise Exception('Error!')
        # todo: 关闭train的开关添加
        self.tred_gnn.train()
        start_time = time.time()
        counter = np.array([0.,0.,0.,0.])
        # for time_stamp in tqdm(range(self.n_layer, self.n_layer + 40), file=sys.stdout):
        for time_stamp in tqdm(range(self.n_layer, self.data.time_length_train), file=sys.stdout, disable=self.opts.disable_bar):
            num_query = self.data.data_splited[time_stamp].shape[0]
            num_batch = num_query // self.batch_size + (num_query % self.batch_size > 0)
            # if self.now_epoch >= 3:
            #     print("2")
            for i in range(num_batch):
                indexes = range(i * self.batch_size, min((i + 1) * self.batch_size, num_query))
                data_batched = torch.tensor(self.data.get_batch(time_stamp, indexes)).cuda()

                self.optimizer.zero_grad()

                scores = self.tred_gnn(time_stamp, data_batched[:, 0], data_batched[:, 1])
                
                loss,valid_num,total_num = self.cal_loss(data_batched, scores)
                loss.backward()
                counter[0]+=total_num
                counter[1]+=valid_num
                counter[2]+=loss.item()
                counter[3]+=(scores!=0).sum()
                self.optimizer.step()
            
                # avoid NaN
                for para in self.tred_gnn.parameters():
                    para_data = para.data.clone()
                    flag = para_data != para_data
                    para_data[flag] = np.random.random()
                    para.data.copy_(para_data)
        counter[1]=counter[1]/counter[0]
        counter[2]=counter[2]/counter[0]
        counter[3]=counter[3]/counter[0]
        # 最终1是覆盖率，0是样本数，2是平均loss,3是平均实体数
        counter = counter.tolist()
        self.loss_history.append(counter[2])
        evaluate_time = time.time()
        self.logger.info(f"Start epoch {self.now_epoch} evaluate")
        v_mrr, v_h1, v_h3, v_h10, v_h100,v_cover,v_loss = self.evaluate()
        nvidia_info = subprocess.run(
            'nvidia-smi', stdout=subprocess.PIPE).stdout.decode()
        with open("memory_info.txt","a")as f:
            print("*"*20,file=f)
            print("pid:",os.getpid(),file=f)
            print(str(self.opts.to_dict()),file=f)
            print(nvidia_info,file=f)
        t_mrr, t_h1, t_h3, t_h10, t_h100,t_cover,t_loss = self.evaluate(data_eval="test")
        finish_time = time.time()
        result = {
            "epoch":self.now_epoch,
            "cover_rate_train":counter[1],
            "loss_train":counter[2],
            "average_entity_num":counter[3],
            "v_mrr":v_mrr,
            "v_h1":v_h1,
            "v_h3":v_h3,
            "v_h10":v_h10,
            "v_h100":v_h100,
            "v_cover":v_cover,
            "v_loss":v_loss,
            "t_mrr":t_mrr,
            "t_h1":t_h1,
            "t_h3":t_h3,
            "t_h10":t_h10,
            "t_h100":t_h100,
            "t_cover":t_cover,
            "t_loss":t_loss,
            "time_train":evaluate_time-start_time,
            "time_valid":finish_time-evaluate_time
        }

        self.train_history.append(result)
        self.logger.info(f"Finish epoch {self.now_epoch}, result:")
        self.logger.info(json.dumps(result))
        if self.best_mrr<v_mrr:
            self.best_mrr = v_mrr
            self.best_result = result
            torch.save(self.tred_gnn,f"{self.result_dir}/best_weight.pt")
            with open(f"{self.result_dir}/best_result.json","w")as f:
                temp = copy(result)
                temp.update(self.opts.to_dict())
                json.dump(temp,f, indent=4, sort_keys=True)
    
    def re_evaluate(self):
        self.logger.info(f"Start re-evaluate")
        if not os.path.exists(f"{self.result_dir}/best_weight.pt"):
            self.logger.info(f"最佳参数文件不存在，请将参数文件'{self.result_dir}/best_weight.pt'放在对应位置")
            return
        
        with open(f"{self.result_dir}/best_weight.pt","rb") as f:
            self.tred_gnn=torch.load(f,map_location=f"cuda:{torch.cuda.current_device()}")
        
        if self.data.time_length_train - self.n_layer < 0:
            raise Exception('Error!')
        start_time = time.time()
        counter = np.array([0.,0.,0.])
        
        # 最终0是覆盖率，1是样本数，2是平均loss
        counter = counter.tolist()
        self.loss_history.append(counter[2])
        evaluate_time = time.time()
        self.logger.info(f"Start epoch {self.now_epoch} evaluate")
        v_mrr, v_h1, v_h3, v_h10, v_h100,v_cover,v_loss = self.evaluate()
        t_mrr, t_h1, t_h3, t_h10, t_h100,t_cover,t_loss = self.evaluate(data_eval="test")
        finish_time = time.time()
        result = {
            "epoch":self.now_epoch,
            "loss_train":counter[2],
            "cover_rate_train":counter[0],
            "v_mrr":v_mrr,
            "v_h1":v_h1,
            "v_h3":v_h3,
            "v_h10":v_h10,
            "v_h100":v_h100,
            "v_cover":v_cover,
            "v_loss":v_loss,
            "t_mrr":t_mrr,
            "t_h1":t_h1,
            "t_h3":t_h3,
            "t_h10":t_h10,
            "t_h100":t_h100,
            "t_cover":t_cover,
            "t_loss":t_loss,
            "time_train":evaluate_time-start_time,
            "time_valid":finish_time-evaluate_time
        }

        self.train_history.append(result)
        self.logger.info(f"Finish reevaluate, result:")
        self.logger.info(json.dumps(result))
        self.logger.info("-"*44)

    def cal_loss(self, data_batched, scores):
        pos_scores = scores[[torch.arange(len(scores)), data_batched[:, 2]]]
        max_n = torch.max(scores, 1, keepdim=True)[0]
        loss = torch.sum(- pos_scores + max_n[:,0] + torch.log(torch.sum(torch.exp(scores - max_n), 1)))
        return loss,pos_scores.nonzero().shape[0],pos_scores.shape[0]
    
    def evaluate_loss(self, data_batched, scores):
        pos_scores = scores[[torch.arange(len(scores)), data_batched[:, 2]]]
        
        max_n = torch.max(scores, 1, keepdim=True)[0]
        loss = torch.sum(- pos_scores + max_n[:,0] + torch.log(torch.sum(torch.exp(scores - max_n), 1)))
        return loss,pos_scores.nonzero().shape[0],pos_scores.shape[0]

    def evaluate_with_multi_steps(self, data_eval='valid'):
        # 此方法测试和验证时，依据不包含正在评估的集合中的数据。验证集依靠训练集的数据，测试集只依靠验证集和测试集的数据
        
        if data_eval == 'valid':
            start_time_stamp = self.data.time_length_train
            end_time_stamp = self.data.time_length_train + self.data.time_length_valid
            evidence_start_point = lambda X:self.data.time_length_train
        elif data_eval == 'test':
            start_time_stamp = self.data.time_length_train + self.data.time_length_valid 
            end_time_stamp = self.data.time_length
            evidence_start_point = lambda X:self.data.time_length_train + self.data.time_length_valid 
        else:
            raise Exception('Error!')
        if start_time_stamp >= end_time_stamp:
            raise Exception('Error!')
        self.logger.info(f"Evaluate {data_eval} data with multi step mode.")
        return self.evaluate_base(start_time_stamp, end_time_stamp, evidence_start_point)
    
    def evaluate_with_single_steps(self, data_eval='valid'):
        # 此方法测试和验证时，依据包含正在评估的集合中的数据。验证集依靠训练集的数据，测试集验证集和测试集的数据
        # 也就是滑动窗口模式
        
        if data_eval == 'valid':
            start_time_stamp = self.data.time_length_train
            end_time_stamp = self.data.time_length_train + self.data.time_length_valid
            get_evidence_start_point = lambda X:X
        elif data_eval == 'test':
            start_time_stamp = self.data.time_length_train + self.data.time_length_valid 
            end_time_stamp = self.data.time_length
            get_evidence_start_point = lambda X:X
        else:
            raise Exception('Error!')
        self.logger.info(f"Evaluate {data_eval} data with single step mode.")
        if start_time_stamp >= end_time_stamp:
            raise Exception('Error!')
        return self.evaluate_base(start_time_stamp, end_time_stamp, get_evidence_start_point)

    def evaluate_base(self, start_time_stamp, end_time_stamp, get_evidence_start_point:Callable):
        self.tred_gnn.eval()
        ranks = []
        counter = np.array([0.,0.,0.])
        # for time_stamp in tqdm(range(start_time_stamp, start_time_stamp + 5), file=sys.stdout):
        for time_stamp in tqdm(range(start_time_stamp, end_time_stamp), file=sys.stdout, disable=self.opts.disable_bar):
            num_query = self.data.data_splited[time_stamp].shape[0]
            num_batch = num_query // self.batch_size + (num_query % self.batch_size > 0)

            for i in range(num_batch):
                indexes = range(i * self.batch_size, min((i + 1) * self.batch_size, num_query))
                data_batched = torch.tensor(self.data.get_batch(time_stamp, indexes)).cuda()
                with torch.no_grad():
                    scores = self.tred_gnn(get_evidence_start_point(time_stamp), data_batched[:, 0], data_batched[:, 1])
                # 观测过拟合程度以及找到的概率
                loss,valid_num,total_num = self.cal_loss(data_batched, scores)
                counter[0] += total_num
                counter[1] += valid_num
                counter[2] += loss.item()
                scores = scores.data.cpu().numpy()
                # 计算rank
                filters_indices = [ (i,obj)  for i,triple in enumerate(data_batched.tolist()) for obj in self.data.time_filter[time_stamp][(triple[0],triple[1])]]
                filters_indices = np.array(filters_indices).T
                filters = np.zeros_like(scores)
                filters[filters_indices[0],filters_indices[1]]=1
                
                rank = utils.cal_ranks(scores, data_batched[:, 2].data.cpu().numpy(),filters)
                ranks = ranks + rank
        counter[1]=counter[1]/counter[0]
        counter[2]=counter[2]/counter[0]
        # 最终1是覆盖率，0是样本数，2是平均loss
        counter = counter.tolist()
        ranks = np.array(ranks)
        mrr, h_1, h_3, h_10, h_100 = utils.cal_performance(ranks)
        return mrr, h_1, h_3, h_10, h_100,counter[1],counter[2]
    
    def process_results(self):
        with open(self.result_dir+"/history.json","w") as f:
            json.dump(self.train_history,f)
        best_result = sorted(self.train_history,key=lambda x:x["v_mrr"],reverse=True)[0]
        # nni.report_final_result(best_result)
        self.logger.info("Finish all epoch, the best is "+str(best_result))
        

class HalfTrainer(Trainer):
    def __init__(self, opts: EnhancedDict | None = None, reevaluate_path: str | None = None):
        super().__init__(opts, reevaluate_path)
        self.scalar = torch.cuda.amp.GradScaler()

    def train_epoch(self):
        self.now_epoch+=1
        self.logger.info(f"Start epoch {self.now_epoch} train")
        if self.data.time_length_train - self.n_layer < 0:
            raise Exception('Error!')
        # todo: 关闭train的开关添加
        self.tred_gnn.train()
        start_time = time.time()
        counter = np.array([0.,0.,0.,0.])
        # for time_stamp in tqdm(range(self.n_layer, self.n_layer + 40), file=sys.stdout):
        for time_stamp in tqdm(range(self.n_layer, self.data.time_length_train), file=sys.stdout, disable=self.opts.disable_bar):
            num_query = self.data.data_splited[time_stamp].shape[0]
            num_batch = num_query // self.batch_size + (num_query % self.batch_size > 0)
            # if self.now_epoch >= 3:
            #     print("2")
            for i in range(num_batch):
                indexes = range(i * self.batch_size, min((i + 1) * self.batch_size, num_query))
                data_batched = torch.tensor(self.data.get_batch(time_stamp, indexes)).cuda()

                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    scores = self.tred_gnn(time_stamp, data_batched[:, 0], data_batched[:, 1])
                    
                    loss,valid_num,total_num = self.cal_loss(data_batched, scores)
                self.scalar.scale(loss).backward()
                self.scalar.step(optimizer=self.optimizer)
                self.scalar.update()
                counter[0]+=total_num
                counter[1]+=valid_num
                counter[2]+=loss.item()
                counter[3]+=(scores!=0).sum()
   
            
                # avoid NaN
                for para in self.tred_gnn.parameters():
                    para_data = para.data.clone()
                    flag = para_data != para_data
                    para_data[flag] = np.random.random()
                    para.data.copy_(para_data)
        counter[1]=counter[1]/counter[0]
        counter[2]=counter[2]/counter[0]
        counter[3]=counter[3]/counter[0]
        # 最终1是覆盖率，0是样本数，2是平均loss,3是平均实体数
        counter = counter.tolist()
        self.loss_history.append(counter[2])
        evaluate_time = time.time()
        self.logger.info(f"Start epoch {self.now_epoch} evaluate")
        v_mrr, v_h1, v_h3, v_h10, v_h100,v_cover,v_loss = self.evaluate()
        nvidia_info = subprocess.run(
            'nvidia-smi', stdout=subprocess.PIPE).stdout.decode()
        with open("memory_info.txt","a")as f:
            print("*"*20,file=f)
            print("pid:",os.getpid(),file=f)
            print(str(self.opts.to_dict()),file=f)
            print(nvidia_info,file=f)
        t_mrr, t_h1, t_h3, t_h10, t_h100,t_cover,t_loss = self.evaluate(data_eval="test")
        finish_time = time.time()
        result = {
            "epoch":self.now_epoch,
            "cover_rate_train":counter[1],
            "loss_train":counter[2],
            "average_entity_num":counter[3],
            "v_mrr":v_mrr,
            "v_h1":v_h1,
            "v_h3":v_h3,
            "v_h10":v_h10,
            "v_h100":v_h100,
            "v_cover":v_cover,
            "v_loss":v_loss,
            "t_mrr":t_mrr,
            "t_h1":t_h1,
            "t_h3":t_h3,
            "t_h10":t_h10,
            "t_h100":t_h100,
            "t_cover":t_cover,
            "t_loss":t_loss,
            "time_train":evaluate_time-start_time,
            "time_valid":finish_time-evaluate_time
        }
        # nni.report_intermediate_result(result)
        self.train_history.append(result)
        self.logger.info(f"Finish epoch {self.now_epoch}, result:")
        self.logger.info(json.dumps(result))
        if self.best_mrr<v_mrr:
            self.best_mrr = v_mrr
            self.best_result = result
            torch.save(self.tred_gnn,f"{self.result_dir}/best_weight.pt")
            with open(f"{self.result_dir}/best_result.json","w")as f:
                temp = copy(result)
                temp.update(self.opts.to_dict())
                json.dump(temp,f, indent=4, sort_keys=True)

    def evaluate_base(self, start_time_stamp, end_time_stamp, get_evidence_start_point:Callable):
        self.tred_gnn.eval()
        ranks = []
        counter = np.array([0.,0.,0.])
        # for time_stamp in tqdm(range(start_time_stamp, start_time_stamp + 5), file=sys.stdout):
        for time_stamp in tqdm(range(start_time_stamp, end_time_stamp), file=sys.stdout, disable=self.opts.disable_bar):
            num_query = self.data.data_splited[time_stamp].shape[0]
            num_batch = num_query // self.batch_size + (num_query % self.batch_size > 0)

            for i in range(num_batch):
                indexes = range(i * self.batch_size, min((i + 1) * self.batch_size, num_query))
                data_batched = torch.tensor(self.data.get_batch(time_stamp, indexes)).cuda()
                with torch.no_grad() and torch.cuda.amp.autocast():
                    scores = self.tred_gnn(get_evidence_start_point(time_stamp), data_batched[:, 0], data_batched[:, 1])
                # 观测过拟合程度以及找到的概率
                loss,valid_num,total_num = self.cal_loss(data_batched, scores)
                counter[0] += total_num
                counter[1] += valid_num
                counter[2] += loss.item()
                scores = scores.data.cpu().numpy()
                # 计算rank
                filters_indices = [ (i,obj)  for i,triple in enumerate(data_batched.tolist()) for obj in self.data.time_filter[time_stamp][(triple[0],triple[1])]]
                filters_indices = np.array(filters_indices).T
                filters = np.zeros_like(scores)
                filters[filters_indices[0],filters_indices[1]]=1
                
                rank = utils.cal_ranks(scores, data_batched[:, 2].data.cpu().numpy(),filters)
                ranks = ranks + rank
        counter[1]=counter[1]/counter[0]
        counter[2]=counter[2]/counter[0]
        # 最终1是覆盖率，0是样本数，2是平均loss
        counter = counter.tolist()
        ranks = np.array(ranks)
        mrr, h_1, h_3, h_10, h_100 = utils.cal_performance(ranks)
        return mrr, h_1, h_3, h_10, h_100,counter[1],counter[2]
    

trainer_dict = {
    "base":Trainer,
    "Half":HalfTrainer
}