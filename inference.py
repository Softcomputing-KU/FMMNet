import warnings
warnings.filterwarnings(action='ignore')
import argparse, cv2
from engine.wrapper import *

import torch, os

import utils.config as config

from glob import glob
from tqdm import tqdm

dataset_name = 'cov19'
# dataset_name = 'mosmed'

from csv2tensor import csv2tensor

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
    dataset = csv2tensor(csv_path=args.test_csv_path, root_path=args.test_root_path, tokenizer=args.bert_type, mode=dataset_name, image_size=[224,224])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # load model        
    model = MMIUNet_Wrapper(args).to(device)
    ckpt = sorted(glob(f'./save_model/{dataset_name}/{args.model_save_filename}.[cC][kK][pP][tT]') \
                + glob(f'./save_model/{dataset_name}/{args.model_save_filename}-v*.[cC][kK][pP][tT]'))[-1]

    print(ckpt)
    checkpoint = torch.load(ckpt, map_location='cpu')["state_dict"]
    model.load_state_dict(checkpoint, strict=True)

    model.eval()

    dir_pth = os.path.join('overlap', dataset_name)
    os.makedirs(os.path.join(dir_pth, 'image'), exist_ok=True)
    os.makedirs(os.path.join(dir_pth, 'gt_overlap'), exist_ok=True)
    os.makedirs(os.path.join(dir_pth, 'pred_overlap', args.model_save_filename), exist_ok=True)
    with torch.no_grad():
        for data in tqdm(dataset, ncols=100, ascii=True):
            input, gt, file_pth, gt_pth, ori_img = data[0], data[1], data[2], data[3], data[4]
            vision, text = input
            vision = vision.to(device)
            ids, mask = text['input_ids'].to(device), text['attention_mask'].to(device)
            text = {'input_ids':ids, 'attention_mask':mask}
            input = [vision, text]

            ori_img, overlap = ori_img.squeeze().numpy().astype('uint8'), (gt > 0.5).squeeze().numpy().astype('uint8')
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR) if ori_img.shape[-1] == 3 else cv2.cvtColor(ori_img, cv2.COLOR_GRAY2BGR)
            y_hat = model(input)
            alpha = 0.15
            pink_color = [180, 105, 255]  # BGR 형식의 분홍색

            overlap = (overlap*255).astype('uint8')
            overlap = cv2.cvtColor(overlap, cv2.COLOR_RGB2BGR) if overlap.shape[-1] == 3 else cv2.cvtColor(overlap, cv2.COLOR_GRAY2BGR)
            overlap[overlap[..., 0] != 0] = pink_color

            # overlap[..., 0] = 0
            overlap = np.where(overlap > 127.5, 255, 0).astype('uint8')
            overlap = cv2.addWeighted(ori_img, 1-alpha, overlap, alpha, 0)

            pred_overlap = (y_hat > 0.5).squeeze().cpu().numpy().astype('uint8')
            pred_overlap = cv2.resize((pred_overlap*255).astype('uint8'), overlap.shape[:2])
            pred_overlap = cv2.cvtColor(pred_overlap, cv2.COLOR_RGB2BGR) if pred_overlap.shape[-1] == 3 else cv2.cvtColor(pred_overlap, cv2.COLOR_GRAY2BGR)
            pred_overlap[pred_overlap[..., 0] != 0] = pink_color

            # pred_overlap[..., 0] = 0
            pred_overlap = np.where(pred_overlap > 127.5, 255, 0).astype('uint8')
            pred_overlap = cv2.addWeighted(ori_img, 1-alpha, pred_overlap, alpha, 0)

            # cv2.imwrite(os.path.join(dir_pth, 'image', os.path.basename(file_pth).split('.')[0]+'.png'), np.transpose(ori_img, (1, 0, 2)))
            # cv2.imwrite(os.path.join(dir_pth, 'gt_overlap', os.path.basename(file_pth).split('.')[0]+'.png'), np.transpose(overlap, (1, 0, 2)))
            cv2.imwrite(os.path.join(dir_pth, 'pred_overlap', args.model_save_filename, os.path.basename(file_pth).split('.')[0]+'.png'), np.transpose(pred_overlap, (1, 0, 2)))