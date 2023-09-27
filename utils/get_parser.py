import os
import glob
import sys
import argparse
import torch
import json
import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging

class JulietArgs:
    def __init__(self, arg_dic) -> None:
        for k,v in arg_dic.items():
            self.__setattr__(k,v)

    

def get_parser(use_local_config=True):
    local_config = None
    if use_local_config:
        with open('config/train_config.json','r') as f:
            local_config=json.load(f)
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True,
                        help='Model name')
    parser.add_argument('-d', '--data_name', type=str, required=True,
                        help='Data name')
    # parser.add_argument("--local_rank", type=int)
    
    args = parser.parse_args()

    dist.init_process_group(backend='nccl', init_method='env://')
    global_rank = dist.get_rank()

    # 计算节点排名和局部排名
    # node_rank = global_rank // 2
    # local_rank = global_rank % 2
    

    
    # rank = torch.distributed.get_rank()
    # print(args.local_rank)
    torch.cuda.set_device(global_rank)
    
    if global_rank == 0:
        print(local_config)
    local_config['model']['model_name'] = args.model_name
    local_config['data']['data_name'] = args.data_name
    # local_config['train']['local_rank'] = local_rank
    
    train_config = JulietArgs(local_config['train'])
    data_config = JulietArgs(local_config['data'])
    model_config = JulietArgs(local_config['model'])

    local_config = JulietArgs({
        "train":train_config,
        "data":data_config,
        "model":model_config,
    })
    return local_config



    
