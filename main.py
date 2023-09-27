import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# train_sampler = DistributedSampler(train_dataset)
# from pytorch_lightning import seed_everything

from utils.utils import *
from utils.get_parser import get_parser
from dataset.Juliet import collate_fn
from scripts.train import train
from diffusers.optimization import get_scheduler
from dataset.Juliet import JulietData
import math
import random
import numpy as np



def seed_everything(seed=1226):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True 

def main():

    
    args = get_parser()
    # args.train.device = "cuda:" + str(args.train.local_rank)  if torch.cuda.is_available() else "cpu"
    # args.train.device = "cuda"  if torch.cuda.is_available() else "cpu"

    seed_everything(args.train.seed)
    rank = torch.distributed.get_rank()
    model_dir = os.path.join("./logs", args.model.model_name)
    checkpoint_dir = os.path.join("./checkpoint_dir", args.model.model_name)
    if rank == 0:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    logger = get_logger(model_dir)
    logger.info(args)
    # dataset
    train_dataset = JulietData(args)   
    sampler = DistributedSampler(train_dataset) 

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=args.data.train_batch_size,
        num_workers=args.data.dataloader_num_workers,
        sampler=sampler
    )

    if rank == 0:
        logger.info("Data Done !!!!!!!!")
    # model
    # 
    model = None
    optims = None
    if args.train.used_pretrain:
        model = get_pretrained_model(args)
        model = model.cuda()
        model = DDP(model,find_unused_parameters=True)
        if rank == 0:
            logger.info("model Done !!!!!!!!")

        optims = get_pretrained_optim(args, model)

        if rank == 0:
            logger.info("optim Done !!!!!!!!")
    elif args.train.load_from_checkpoint:
        model = get_checkpoint_model(args)
        optims = get_checkpoint_optim(args)
    else:
        model = get_init_model(args)
        optims = get_init_optim(args, model)
    
    # schedulers = None
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.train.gradient_accumulation_steps)
    if args.train.max_train_steps is None:
        args.train.max_train_steps = args.train.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        args.train.lr_scheduler,
        optimizer=optims,
        num_warmup_steps=args.train.lr_warmup_steps,
        num_training_steps=args.train.max_train_steps,
    )
    # train

    train(
        args = args,
        rank = rank,
        model = model,
        dataloders = [train_dataloader,None],
        optims = optims,
        schedulers = lr_scheduler,
        logger = logger,
        )

if __name__ == '__main__':
    main()