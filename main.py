import base_model
from utils import EnhancedDict,gpu_setting
import json
import argparse

def main(initial_dict:dict):
    opts = EnhancedDict(initial_dict)
    path = './data/' + opts.dataset + '/'
    
    # opts.model_name = "TRED_GNN4"
    # opts.dataset=dataset
    opts.path = path

    # opts.n_layer = n_layer
    # opts.batch_size = batch_size

    opts.disable_bar = False
    opts.tag = f"L{opts.n_layer}"+opts.tag
    
    # 自动选择合适的GPU
    gpu_setting(opts.get("gpu",-1))
    if opts.train_mode.lower() == 'base':
        trainer = base_model.Trainer(opts)
    elif opts.train_mode.lower() == 'half':
        trainer = base_model.HalfTrainer(opts)
    else:
        raise Exception("未实现的训练模式")
    for epoch in range(opts.epochs):
        trainer.train_epoch()
        """if epoch > 0:
            if model.train_history[-1][1] < model.train_history[-2][1]:
                decline_step = decline_step + 1
            else:
                decline_step = 0
            if decline_step >= stop_step:
                print('best : mrr ',model.train_history[-stop_step][1],' hist@1 ',model.train_history[-stop_step][2],' hist@10 ',model.train_history[-stop_step][3])
                break"""
    trainer.process_results()

    

if __name__ == '__main__':
    # 1. 定义命令行解析器对象
    parser = argparse.ArgumentParser(description='Demo of argparse')
    
    # 2. 添加命令行参数
    parser.add_argument('--n_layer', type=int, default=3, help="经过几次GNN计算，在子图中的搜索深度")
    parser.add_argument('--dataset', type=str, default="ICEWS14s", help="数据集的名字，需要和data目录下的数据集文件夹名字匹配")
    parser.add_argument('--batch_size', type=int, default=128, help="略")
    parser.add_argument('--single_timestamp_layer_numbers', type=int, default=2, help="对20无效")
    parser.add_argument('--gnn_mode', type=str, default="unique" , help="对20无效")
    parser.add_argument('--time_mode', type=str, default="embedding", help="对20无效")
    parser.add_argument('--model_name', type=str, default="TRED_GNN", help="使用的模型型号")
    parser.add_argument('--window_size', type= int, default=10, help="搜索的近期子图的时间范围（局部规则）")
    parser.add_argument('--hidden_dim', type= int, default=64)
    parser.add_argument('--max_global_window_size', type= int, default=5000, help="全局规则的检索时间片范围")
    parser.add_argument('--epochs', type= int, default=20)
    parser.add_argument('--gpu', type= int, default=-1, help="使用哪一张显卡，默认为-1，自动选择显存占用最低的显卡，其他情况下选择对应标号的卡")
    parser.add_argument('--tag', type= str, default='', help="tag标记，用于细节不同的模型")
    parser.add_argument('--train_mode', type= str, default='base', help="默认为base，即使用float32精度训练，推荐使用half，混合精度训练。目前混合精度只适配了20号模型")
    parser.add_argument('--lr', type= float, default=0.005)
    parser.add_argument('--attention_dim', type= int, default=5)
    parser.add_argument('--act', type= str, default="idd", choices=['idd', 'relu', 'tanh'])
    parser.add_argument('--lamb', type= float, default=0.00012)
    parser.add_argument('--dropout', type= float, default=0.25)
    parser.add_argument('--time_dim', type=int, default=16, help="时间维度的大小")
    # 3. 从命令行中结构化解析参数
    args = parser.parse_args()
    main(initial_dict=vars(args))
    