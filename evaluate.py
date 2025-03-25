import warnings
warnings.filterwarnings(action='ignore')
import argparse
from engine.wrapper import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl  
from pytorch_lightning.utilities.model_summary import ModelSummary

from utils.dataset import QaTa, MosMed
import utils.config as config

from glob import glob

dataset_name = 'cov19'
# dataset_name = 'mosmed'

def get_parser():
    parser = argparse.ArgumentParser(
        description='Language-guide Medical Image Segmentation')
    parser.add_argument('--config',
                        default='./config/training_cov19.yaml' if dataset_name == 'cov19' else './config/training_mosmed.yaml',
                        type=str,
                        help='config file')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)

    return cfg

if __name__ == '__main__':

    args = get_parser()
    # load model        
    model = MMIUNet_Wrapper(args)
    summary = ModelSummary(model)
    print(summary)

    ckpt_list = sorted(glob(f'./save_model/{dataset_name}/{args.model_save_filename}.[cC][kK][pP][tT]') \
                + glob(f'./save_model/{dataset_name}/{args.model_save_filename}-v*.[cC][kK][pP][tT]'))

    for ckpt in ckpt_list:
        print(ckpt)
        checkpoint = torch.load(ckpt, map_location='cpu')["state_dict"]
        model.load_state_dict(checkpoint, strict=True)
        
        # dataloader
        ds_test = QaTa(csv_path=args.test_csv_path,
                    root_path=args.test_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='test') if dataset_name == 'cov19' else \
                MosMed(csv_path=args.test_csv_path,
                    root_path=args.test_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='test')
        dl_test = DataLoader(ds_test, batch_size=args.valid_batch_size, shuffle=False, num_workers=8)

        trainer = pl.Trainer(accelerator='gpu',
                            gpus=[0] if dataset_name == 'cov19' else [1],
                            # gpus=[2] if dataset_name == 'cov19' else [3],
                            devices=1
                            ) 
        model.eval()

        trainer.test(model, dl_test) 