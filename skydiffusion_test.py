import os
import torch
import argparse
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from share import *
from skydiffusion_dataset import SkyDataset
from cldm.logger import ValidImageLogger
from cldm.model import create_model, load_state_dict


def create_argparser():
    defaults = dict(
        seed = 42,
        config_path = './models/lacldm_v15.yaml',
        model_path = './models/lacontrol_sd15_ini.ckpt',
        image_width = 512,
        image_height = 512,
        text_prompt = '',
        data_file_path = './test.csv',
        batch_size = 4,
        result_dir = './results',
        logger_freq = 1,
        sample_num = 10000,
        unconditional_guidance_scale = 9.0,
        num_gpus = 8,
        is_cvusa = False,
        negative_prompt = ""
        dataset_name = "CVACT"
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    pl.seed_everything(args.seed, workers=True)

    model = create_model(args.config_path).cpu()
    model.load_state_dict(load_state_dict(args.model_path, location='cpu'), strict=False)
    model = model.cuda()
    model.eval()

    image_size = (args.image_width, args.image_height)
    print(args.data_file_path)
    print(args.is_cvusa)
    print('neg_prompt', args.negative_prompt)
    print('args.resultdir:',args.result_dir)
    dataset = SkyDataset(args.data_file_path, args.is_cvusa, args.text_prompt, image_size, dataset_name=args.dataset_name)
    dataloader = DataLoader(dataset, num_workers=4, batch_size=args.batch_size, shuffle=False)

    log_images_kwargs = {
        'unconditional_guidance_scale': args.unconditional_guidance_scale,
        'negative_prompt': args.negative_prompt
    }
    logger = ValidImageLogger(args.result_dir, args.logger_freq, log_images_kwargs=log_images_kwargs)
    trainer = pl.Trainer(
        gpus=args.num_gpus,
        strategy="ddp",
        precision=32,
        max_epochs=1,
        callbacks=[logger]
    )
    trainer.validate(model, dataloader)


if __name__ == '__main__':
    main()
