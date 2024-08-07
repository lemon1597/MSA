import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import random_box, random_click

from utils import *

class ISIC2016(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):

        df = pd.read_csv(os.path.join(data_path + '/', 'ISBI2016_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist() # 第 1 列所有行
        self.label_list = df.iloc[:,2].tolist() # 第 2 列所有行
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        # self.img_size = args['image_size']

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask) / 255, point_label)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask).int()
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask
        name = name.split('\\')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }


# def test_get_item():
#
#     args = {
#         'image_size' : 1024,
#         'out_size' : 256,
#         'data_path' : 'E:\deeplearningprojects\codes\Medical-SAM-Adapter\data\isic'
#     }
#
#     transform_train = transforms.Compose([
#         transforms.Resize((args['image_size'],args['image_size'])),
#         transforms.ToTensor(),
#     ])
#
#     transform_train_seg = transforms.Compose([
#         transforms.Resize((args['out_size'],args['out_size'])),
#         transforms.ToTensor(),
#     ])
#
#     # 实例化数据集
#     data = ISIC2016(args, args['data_path'], transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
#
#     item = data[0]
#
#     print('image:', item['image'])
#     print('label:', item['label'])
#     print('p_label:', item['p_label'])
#     print('pt:', item['pt'])
#     print('image_meta_dict:', item['image_meta_dict'])
#
# if __name__ == "__main__":
#     test_get_item()