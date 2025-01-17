import os
import pickle

# import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from utils import generate_click_prompt, random_box, random_click


class ToothFairy(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):


        self.args = args
        self.data_path = os.path.join(data_path,'Dataset')
        self.name_list = os.listdir(self.data_path)
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)


    def __getitem__(self, index):
        point_label = 1


        """Get the images"""
        name = self.name_list[index]
        img = np.load(os.path.join(self.data_path,name,'data.npy'))
        mask = np.load(os.path.join(self.data_path,name,'gt_sparse.npy'))

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])
        img = np.transpose(img,(1,2,0))
        mask = np.transpose(mask,(1,2,0))

        # img = np.resize(mask,(self.args.image_size, self.args.image_size,img.shape[-1]))
        # mask = np.resize(mask,(self.args.out_size,self.args.out_size,mask.shape[-1]))

        img = np.resize(img,(self.args.image_size, self.args.image_size,img.shape[-1]))
        mask = np.resize(mask,(self.args.out_size,self.args.out_size,mask.shape[-1]))

        img = torch.tensor(img).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)
        mask = torch.clamp(mask,min=0,max=1).int()

        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask), point_label)
        # if self.transform:
        #     state = torch.get_rng_state()
        #     img = self.transform(img)
        #     torch.set_rng_state(state)

        #     if self.transform_msk:
        #         mask = self.transform_msk(mask)
                
        #     # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
        #     #     mask = 1 - mask
        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }


# def test_get_item(data_path):
#     # 假设args是一个包含所需参数的对象，这里我们使用字典来模拟
#     args = {
#         'image_size': 256,  # 假设的图像尺寸
#         # 其他参数...
#     }
#
#     # 实例化数据集类
#     dataset = ToothFairy(args, data_path, mode='Training', prompt='click')
#
#     # 测试检索数据集中的第一个项目
#     if len(dataset) > 0:
#         item = dataset[0]
#         print("Test item retrieved:")
#         print(f"Image tensor shape: {item['image'].shape}")
#         print(f"Mask tensor shape: {item['label'].shape}")
#         print(f"Point label: {item['p_label']}")
#         print(f"Points: {item['pt']}")
#         print(f"Image metadata: {item['image_meta_dict']}")
#     else:
#         print("The dataset is empty.")
#
#     return item
#
# # 用于测试的主函数
# def main():
#     data_path = r'C:\Users\linmeng\Downloads\ToothFairy_Dataset'  # 替换为你的数据集路径
#     test_get_item(data_path)
#
# if __name__ == "__main__":
#     main()