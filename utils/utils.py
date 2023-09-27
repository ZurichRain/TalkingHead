import os
import glob
import sys
import argparse
import torch
import json
import logging

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from models.Juliet import Juliet

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging

def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger

def get_pretrained_model(args):
    # 加载预训练模型
    # checkpoint_ = torch.load(args.pretrained_model_name_or_path, map_location='cpu')
    noise_scheduler = DDPMScheduler.from_pretrained(args.model.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.model.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
            args.model.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.model.revision
        )
    
    audio_encoder = None
    
    vae = AutoencoderKL.from_pretrained(
        args.model.pretrained_model_name_or_path, subfolder="vae", revision=args.model.revision
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        args.model.pretrained_model_name_or_path, subfolder="unet", revision=args.model.non_ema_revision
    )
    # print(unet.config)
    # exit()
    
    model = Juliet(noise_scheduler=noise_scheduler,
                   unet=unet,
                   vae=vae,
                   model_args= args,
                   text_encoder=text_encoder)
    return model

def get_init_optim(args,model):

    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        model.module.unet.parameters(),
        lr=args.train.learning_rate,
        betas=(args.train.adam_beta1, args.train.adam_beta2),
        weight_decay=args.train.adam_weight_decay,
        eps=args.train.adam_epsilon,
    )
    
    return optimizer

def get_pretrained_optim(args, model):
    
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        model.module.unet.parameters(),
        lr=args.train.learning_rate,
        betas=(args.train.adam_beta1, args.train.adam_beta2),
        weight_decay=args.train.adam_weight_decay,
        eps=args.train.adam_epsilon,
    )
    
    return optimizer


def save_checkpoint(model, global_step,args):
    save_path = os.path.join(args.train.output_dir, args.model.model_name+"/"+f"checkpoint-{global_step}")
    torch.save(model.module.state_dict(), save_path)
