import os
import torch
import pandas as pd
from monai.transforms import (Compose, NormalizeIntensityd, Resized, ToTensord, LoadImaged, EnsureChannelFirstd)
from transformers import AutoTokenizer
from tqdm import tqdm

def csv2tensor(csv_path=None, root_path=None, tokenizer=None, mode='cov19', image_size=[224,224]):
    with open(csv_path, 'r') as f:
        data = pd.read_csv(f)
    image_list = list(data['Image'])
    caption_list = list(data['Description'])
    print(len(image_list))

    root_path = root_path
    image_size = image_size

    tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    trans_ori = transform(True, image_size)
    trans = transform(False, image_size)

    total_len = len(image_list)

    outputs = []

    for idx in tqdm(range(total_len), ncols=100):
        image_pth = os.path.join(root_path,'Images',image_list[idx].replace('mask_',''))
        image = image_pth

        gt_pth = os.path.join(root_path, 'GTs', image_list[idx])
        gt = gt_pth

        caption = caption_list[idx]
        token_output = tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=24, 
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token, mask = token_output['input_ids'], token_output['attention_mask']

        data = {'image':image, 'gt':gt, 'token':token, 'mask':mask}
        ori_image = trans_ori(data)
        data = trans(data)

        image, gt, token, mask = data['image'], data['gt'], data['token'], data['mask']
        gt = torch.where(gt==255, 1, 0)
        gt = (torch.sum(gt, axis=0)/1.0).int() if mode=='cov19' else (torch.sum(gt, axis=0)/3.0).int()
        gt = torch.unsqueeze(gt, 0)
        # text = {'input_ids':token.squeeze(dim=0), 'attention_mask':mask.squeeze(dim=0)}
        text = {'input_ids':token, 'attention_mask':mask}

        image = torch.unsqueeze(image, 0)

        outputs.append(([image, text], ori_image['gt'], image_pth, gt_pth, ori_image['image']))

    return outputs

def transform(only_load = False, image_size=[224,224]):
    trans = Compose([
        LoadImaged(["image","gt"], reader='PILReader'),
        EnsureChannelFirstd(["image","gt"]),
        Resized(["image"],spatial_size=image_size,mode='bicubic'),
        Resized(["gt"],spatial_size=image_size,mode='nearest'),
        NormalizeIntensityd(['image'], channel_wise=True),
        ToTensord(["image","gt","token","mask"])
    ]) if only_load == False else LoadImaged(["image","gt"], reader='PILReader')

    return trans