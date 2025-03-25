import warnings
warnings.filterwarnings(action='ignore')
import torch
from torch.utils.data import DataLoader
from utils.dataset import QaTa, MosMed
import utils.config as config
# from torch.optim import lr_scheduler
from engine.wrapper import *

import pytorch_lightning as pl    
# from torchmetrics import Accuracy, Dice
# from torchmetrics.classification import BinaryJaccardIndex
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse


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
    print("cuda:",torch.cuda.is_available())
    
    sub_string = args.model_save_filename[-2:]
    sample_rate = float(sub_string) * .01 if sub_string.isdigit() else 1

    ds_train = QaTa(csv_path=args.train_csv_path,
                    root_path=args.train_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    sample_ratio=sample_rate,
                    mode='train') if dataset_name == 'cov19' else \
               MosMed(csv_path=args.train_csv_path,
                    root_path=args.train_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    sample_ratio=sample_rate,
                    mode='train')

    ds_valid = QaTa(csv_path=args.valid_csv_path,
                    root_path=args.valid_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='valid') if dataset_name == 'cov19' else \
               MosMed(csv_path=args.valid_csv_path,
                    root_path=args.valid_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='valid')

    dl_train = DataLoader(ds_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.train_batch_size)
    dl_valid = DataLoader(ds_valid, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.valid_batch_size)
    
    if args.model_save_filename == "medseg":
        model = LanGuideMedSegWrapper(args)

    elif "mmiunet" in args.model_save_filename:
        model = MMIUNet_Wrapper(args)

    # elif args.model_save_filename == "mmiunet":
    #     model = MMIUNet_Wrapper(args)

    elif args.model_save_filename == "mmiunext_s":
        model = MMIUNeXt_S_Wrapper(args)
        
    # else:
    #     model = MMIUNet_GS_Wrapper(args)

    ## 1. setting recall function
    model_ckpt = ModelCheckpoint(
        dirpath=args.model_save_path,
        filename=args.model_save_filename,
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        verbose=True,
    )

    early_stopping = EarlyStopping(monitor = 'val_loss',
                            patience=args.patience,
                            mode = 'min'
    )

    logger = CSVLogger(save_dir=args.model_save_path, name=args.model_save_filename)

    ## 2. setting trainer
    trainer = pl.Trainer(logger=logger,#True,
                         min_epochs=args.min_epochs,max_epochs=args.max_epochs,
                         accelerator='gpu',
                         devices=args.device,
                         callbacks=[model_ckpt],#, early_stopping],
                         enable_progress_bar=True,
                        #  gpus=[0] if dataset_name == 'cov19' else [1]
                         gpus=[2] if dataset_name == 'cov19' else [3]
                         # enable_progress_bar=False,
                        #  gpus=[0,1,2,3],
                        #  strategy='ddp',
                        #  precision=16
                        ) 

    ## 3. start training
    print('start training')
    trainer.fit(model, dl_train, dl_valid)
    print('done training')
    
    # from pytorch_lightning.utilities.model_summary import ModelSummary
    # summary = ModelSummary(model)
    # print(summary)

    # from thop import profile
    # flops, params = profile(model, [torch.randn(1,1,224,224), {'input_ids':torch.randn(1,24), 'attention_mask':torch.randn(1,24)}])
    # flops, params = profile(model, [torch.randn(1,224,224), {'input_ids':torch.randn(24), 'attention_mask':torch.randn(24)}])